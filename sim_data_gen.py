from typing import NamedTuple

import numpy as np
import pandas as pds
import torch
from sklearn.decomposition import PCA

from torch.utils.data import TensorDataset
        
from global_config import DEVICE, DTYPE
from utils import bootstrap_RMSE


def logistic(x):
    return 1.0 / (1.0 + np.exp(-1.0 * x))


def cause_to_num(cause):
    # convert the causes into an index
    weight_vector = np.power(2, np.arange(0, cause.shape[1]))
    cause_ind = np.matmul(cause, weight_vector)
    return cause_ind


# def num_to_cause(num, n_treatment):
#     num = torch.tensor(num)
#     cause = num.unsqueeze(-1).to(torch.long)
#     cause_mat = torch.zeros(cause.shape[0], n_treatment)
#     cause_mat.scatter_(1, cause, 1)
#


def dcg_at_k(r, k):
    r = r[:k]
    return np.sum(r / np.log2(np.arange(2, r.size + 2)))


def ndcg_at_k(r, k):
    # ndcg_at_k(score_mat, k)
    dcg_max = dcg_at_k(np.array(sorted(r[0], reverse=True)), k)

    res = []
    for i in range(r.shape[0]):
        res.append(dcg_at_k(r[i], k) / dcg_max)
    return np.array(res)


class DataGeneratorConfig(NamedTuple):
    n_confounder: int
    n_cause: int
    n_outcome: int
    sample_size: int
    p_confounder_cause: float
    p_cause_cause: float
    cause_noise: float
    outcome_noise: float
    linear: bool
    sim_id: str
    p_outcome_single: float = 1.0
    p_outcome_double: float = 0.0
    train_frac: float = 0.7
    val_frac: float = 0.1
    n_flip: int = 1
    confounding_level: float = 1.0
    real_data: bool = False
    outcome_interaction: bool = False
    sample_size_train: int = 0


class DataGenerator:
    def __init__(
        self,
        n_confounder,
        n_cause,
        n_outcome,
        sample_size,
        p_confounder_cause,
        p_cause_cause,
        cause_noise,
        outcome_noise,
        linear,
        p_outcome_single=1,
        p_outcome_double=1,
        confounding_level=1.0,
        train_frac=0.7,
        val_frac=0.1,
        real_data=False,
        outcome_interaction=False,
        outcome_interaction3=False,
        no_confounder=False,
        device=DEVICE,
        dtype=DTYPE,
    ):
        assert train_frac + val_frac <= 1
        self.n_confounder = n_confounder
        self.n_cause = n_cause
        self.n_outcome = n_outcome
        self.sample_size = sample_size
        self.p_confounder_cause = p_confounder_cause
        self.p_cause_cause = p_cause_cause
        self.cause_noise = cause_noise
        self.outcome_noise = outcome_noise
        self.linear = linear
        self.outcome_interaction = outcome_interaction
        self.outcome_interaction3 = outcome_interaction3
        self.p_outcome_single = p_outcome_single
        self.p_outcome_double = p_outcome_double
        self.confounder = None
        self.coefficient_confounder_cause = None
        self.coefficient_cause_cause = None
        self.coefficient_cause_outcome = None
        self.coefficient_confounder_outcome = None
        self.cause = None
        self.outcome = None
        self.outcome_error = None
        self.cause_error = None
        self.cause_logit = None
        self.train_size = int(train_frac * sample_size)
        self.val_size = int(val_frac * sample_size)
        self.device = device
        self.dtype = dtype
        self.confounding_level = confounding_level
        self.generated = False
        self.descendant = None
        self.real_data = real_data
        self.coef_list = None
        self.outcome_list = None
        self.cause_ind = None
        self.ranks = None
        self.interaction_coef = None
        self.pca = None
        self.no_confounder = no_confounder

        # iterate over causes
        n_treatment = int(np.power(2, self.n_cause))
        # 2^K, K
        cause_ref = np.zeros((n_treatment, self.n_cause))
        for i in range(n_treatment):
            tmp = i
            for j in range(1, self.n_cause + 1):
                value = tmp % 2
                tmp = tmp // 2
                cause_ref[i, j - 1] = value

        self.cause_ref = cause_ref
        self.n_treatment = n_treatment

    def _make_tensor(self, x):
        if type(x) == np.ndarray:
            return torch.tensor(x, dtype=self.dtype, device=self.device)  # pylint: disable=not-callable
        else:
            return x.to(dtype=self.dtype, device=self.device)

    def get_coefficient_confounder_cause(self):
        coef = np.random.randn(self.n_confounder, self.n_cause)
        mask = np.random.binomial(1, self.p_confounder_cause, (self.n_confounder, self.n_cause))
        if not self.no_confounder:
            self.coefficient_confounder_cause = coef * mask
        else:
            self.coefficient_confounder_cause = coef * 0.0

    def get_coefficient_cause_cause(self):
        coef = np.random.randn(self.n_cause, self.n_cause)
        mask = np.random.binomial(1, self.p_cause_cause, (self.n_cause, self.n_cause))
        self.coefficient_cause_cause = coef * mask

        children_list = [list() for i in range(self.n_cause)]
        for i in reversed(range(self.n_cause)):
            children = list(np.where(mask[i, min(i + 1, self.n_cause) :])[0] + i + 1)

            while True:
                set_card = len(children)
                for j in children:
                    children = children + children_list[j]

                children = list(set(children))
                new_set_card = len(children)
                if new_set_card == set_card:
                    break

            children_list[i] = children

        self.descendant = children_list

    def get_coefficient_outcome(self):
        coef_cause = np.random.randn(self.n_cause)
        coef_confounder = np.random.randn(self.n_confounder)
        self.coefficient_cause_outcome = coef_cause
        if not self.no_confounder:
            self.coefficient_confounder_outcome = coef_confounder
        else:
            self.coefficient_confounder_outcome = coef_confounder * 0.0

    def generate_confounder(self):
        """
        Generate the confounders using i.i.d Gaussian(0, 1). Sample size x dim
        """
        if not self.no_confounder:
            self.confounder = np.random.randn(self.sample_size, self.n_confounder)
        else:
            self.confounder = np.zeros((self.sample_size, self.n_confounder)) * 1.0

    def get_one_cause(self, i, confounder_factor_mean, previous_causes):
        # K
        mean = confounder_factor_mean[:, i]

        # todo: add parameter here
        if i > 0:
            cause_factor_mean = (
                np.matmul(previous_causes, self.coefficient_cause_cause[:i, i]) / previous_causes.shape[1]
            )
        else:
            cause_factor_mean = 0
        cause_error = np.random.randn(self.sample_size) * self.cause_noise
        cause_logit = mean + cause_factor_mean + cause_error
        cause = np.random.binomial(1, logistic(cause_logit))
        return cause, cause_logit, cause_error

    def generate_cause(self):
        confounder_factor_mean = (
            np.matmul(self.confounder, self.coefficient_confounder_cause) / self.n_confounder * self.confounding_level
        )

        cause_list = []
        cause_logit_list = []
        cause_error_list = []
        for i in range(self.n_cause):
            if i > 0:
                previous_causes = np.stack(cause_list, axis=-1)
            else:
                previous_causes = None

            new_cause, cause_logit, cause_error = self.get_one_cause(i, confounder_factor_mean, previous_causes)
            cause_list.append(new_cause)
            cause_logit_list.append(cause_logit)
            cause_error_list.append(cause_error)

        self.cause = np.stack(cause_list, axis=-1)
        self.cause_logit = np.stack(cause_logit_list, axis=-1)
        self.cause_error = np.stack(cause_error_list, axis=-1)

    def generate_outcome(self):
        if not self.outcome_interaction:
            # no interaction
            outcome = np.matmul(self.cause, self.coefficient_cause_outcome)
            outcome = outcome + np.matmul(self.confounder, self.coefficient_confounder_outcome)

        elif self.outcome_interaction3:
            # three way interaction
            ones = np.ones((self.sample_size, 1))
            outcome = np.zeros(self.sample_size)
            feat = np.concatenate((ones, self.confounder, self.cause), axis=1)
            ncol_feat = feat.shape[1]

            coef_list = []

            for i in range(ncol_feat):
                for j in range(i, ncol_feat):
                    for k in range(j, ncol_feat):
                        for h in range(k, ncol_feat):

                            feat_in = feat[:, i] * feat[:, j] * feat[:, k] * feat[:, h]
                            coef = np.random.randn()

                            if i == 0:
                                mask = np.random.binomial(1, self.p_outcome_single)
                            else:
                                mask = np.random.binomial(1, self.p_outcome_double)
                            coef *= mask
                            coef_list.append(coef)

                            outcome = outcome + feat_in * coef

            self.interaction_coef = np.array(coef_list)

        else:
            # two way interaction
            ones = np.ones((self.sample_size, 1))
            outcome = np.zeros(self.sample_size)
            feat = np.concatenate((ones, self.confounder, self.cause), axis=1)
            ncol_feat = feat.shape[1]

            coef_list = []

            for i in range(ncol_feat):
                for j in range(i, ncol_feat):
                    feat_in = feat[:, i] * feat[:, j]
                    coef = np.random.randn()

                    if i == 0:
                        mask = np.random.binomial(1, self.p_outcome_single)
                    else:
                        mask = np.random.binomial(1, self.p_outcome_double)
                    coef *= mask
                    coef_list.append(coef)

                    outcome = outcome + feat_in * coef

            self.interaction_coef = np.array(coef_list)

        self.outcome_error = np.random.randn(self.sample_size) * self.outcome_noise
        outcome = outcome + self.outcome_error
        outcome = outcome[:, None]
        assert len(outcome.shape) == 2
        if not self.linear:
            if not self.outcome_interaction:
                outcome = logistic(outcome / np.sqrt(self.n_confounder))
            else:
                outcome = logistic((outcome - np.mean(outcome)) / np.std(outcome))
        self.outcome = outcome

    def generate_counterfactual(self, new_cause):
        if not self.outcome_interaction:
            outcome = np.matmul(new_cause, self.coefficient_cause_outcome)
            outcome = outcome + np.matmul(self.confounder, self.coefficient_confounder_outcome)

        elif self.outcome_interaction3:

            # two way interaction
            ones = np.ones((self.sample_size, 1))
            outcome = np.zeros(self.sample_size)
            feat = np.concatenate((ones, self.confounder, new_cause), axis=1)
            ncol_feat = feat.shape[1]
            counter = 0

            for i in range(ncol_feat):
                for j in range(i, ncol_feat):
                    for k in range(j, ncol_feat):
                        for h in range(k, ncol_feat):

                            feat_in = feat[:, i] * feat[:, j] * feat[:, k] * feat[:, h]
                            outcome = outcome + feat_in * self.interaction_coef[counter]
                            counter += 1

        else:
            ones = np.ones((self.sample_size, 1))
            outcome = np.zeros(self.sample_size)
            feat = np.concatenate((ones, self.confounder, new_cause), axis=1)
            ncol_feat = feat.shape[1]

            counter = 0
            for i in range(ncol_feat):
                for j in range(i, ncol_feat):
                    feat_in = feat[:, i] * feat[:, j]
                    outcome = outcome + feat_in * self.interaction_coef[counter]
                    counter += 1

        # use the old error
        outcome = outcome + self.outcome_error
        outcome = outcome[:, None]
        assert len(outcome.shape) == 2
        if not self.linear:
            outcome = logistic(outcome / np.sqrt(self.n_confounder))
        return outcome

    def generate_real_outcome(self, cause, confounder, noise=0.0):
        # feature dimension
        ones = np.ones((confounder.shape[0], 1))
        feat_dim = self.n_cause + self.n_confounder + 1
        coef_list = []
        for i in range(feat_dim):
            for j in range(i + 1, feat_dim):
                this_coef = np.random.randn() * 0.5
                coef_list.append(this_coef)
        self.coef_list = coef_list

        # iterate over causes
        n_treatment = int(np.power(2, self.n_cause))
        # 2^K, K
        cause_ref = np.zeros((n_treatment, self.n_cause))
        for i in range(n_treatment):
            tmp = i
            for j in range(1, self.n_cause + 1):
                value = tmp % 2
                tmp = tmp // 2
                cause_ref[i, j - 1] = value

        self.cause_ref = cause_ref
        # 2^K
        cause_ind = cause_to_num(cause_ref)

        # generate outcomes for all causes
        outcome_list = []
        for x in range(n_treatment):
            # 1, K
            cause_setting = cause_ref[x : x + 1, :]
            # N, K
            cause_sample = np.concatenate([cause_setting] * confounder.shape[0], axis=0)
            # N, K + D + 1
            feat = np.concatenate([confounder, cause_sample, ones], axis=-1)

            outcomes_mean = np.zeros((cause.shape[0], 1))
            loc = 0
            for i in range(feat.shape[1]):
                for j in range(i + 1, feat.shape[1]):
                    this_feat = feat[:, i : i + 1] * feat[:, j : j + 1]
                    this_coef = coef_list[loc]
                    loc += 1
                    outcomes_mean = outcomes_mean + this_feat * this_coef

            this_outcome = np.abs(outcomes_mean) + noise * np.random.exponential(1.0, outcomes_mean.shape)

            assert this_outcome.shape[0] == cause.shape[0]
            assert this_outcome.shape[1] == 1
            outcome_list.append(this_outcome)

        self.outcome_list = outcome_list
        self.cause_ind = cause_ind

        # generate true ranking
        # smaller, better
        # N, 2^K
        outcome_tensor = np.concatenate(self.outcome_list, axis=-1)
        order = outcome_tensor.argsort(axis=-1)
        # N, 2^K
        ranks = order.argsort(axis=-1)
        self.ranks = ranks

        # select observed outcomes from the list
        # N
        obs_cause_ind = cause_to_num(cause)
        outcomes = np.zeros((cause.shape[0], 1))
        for i in range(cause.shape[0]):
            location = np.where(cause_ind == obs_cause_ind[i])[0][0]
            outcomes[i, 0] = self.outcome_list[location][i, 0]

        return outcomes

    def get_relevance_score(self, predicted_ranking):
        # predicted ranking on the testing set
        n_treatment = np.power(2, self.n_cause)

        score_list = []
        for i in range(predicted_ranking.shape[0]):
            predicted = predicted_ranking[i]
            true = self.ranks[self.train_size + self.val_size + i]
            # scores = n_treatment - true[predicted.astype('int')]
            scores = (true[predicted.astype("int")] < 5).astype("float")
            score_list.append(scores)

        # N, 2^K
        score_mat = np.stack(score_list, axis=0)
        return score_mat

    def generate_oracle_improvements(self):
        assert self.generated
        outcome_tensor = np.concatenate(self.outcome_list, axis=-1)
        best_outcome = np.min(outcome_tensor, axis=-1)
        improvements = self.outcome - best_outcome[:, None]
        return improvements[self.train_size + self.val_size :, 0]

    def get_predicted_improvements(self, order, k=5):
        pred_list = []
        for j in range(k):
            best_y_ind = order[:, j]
            score_list = []
            for i in range(best_y_ind.shape[0]):
                predicted = best_y_ind[i]
                true = self.outcome_list[predicted][self.train_size + self.val_size + i]
                score_list.append(true)

            pred = np.array(score_list)
            pred_list.append(pred)

        pred_mat = np.stack(pred_list, axis=-1)
        pred = np.mean(pred_mat, axis=-1)
        improvements = self.outcome - pred[:, None]
        return improvements[:, 0]

    def generate_real(self):
        """
        读取 real_data/{confounder,cause,outcome}.csv，自适应列数与样本数；
        若 config.sample_size < N，则稳定随机下采样；按 7/1/2（或传入的 train_frac/val_frac）切分；
        并缓存 train/val/test 的 confounder/cause/outcome，供 generate_dataset 与 generate_test_real 使用。
        """
        import numpy as np
        import torch
        import pandas as pds
        from torch.utils.data import TensorDataset

        # 1) 读入三张 CSV
        conf = pds.read_csv("real_data/confounder.csv").values  # [N, Z]
        cause = pds.read_csv("real_data/cause.csv").values      # [N, T]
        out   = pds.read_csv("real_data/outcome.csv").values    # [N, Y]

        # 2) 形状检查
        assert conf.ndim == 2 and cause.ndim == 2 and out.ndim == 2, "CSV 应为二维表（需包含表头）"
        N, Z = conf.shape
        Nt, T = cause.shape
        No, Y = out.shape
        assert Nt == N and No == N, f"conf/cause/outcome 行数需一致：conf={N}, cause={Nt}, outcome={No}"

        # 3) 自适应填充配置（当为 0/None 时）
        if not getattr(self, "n_confounder", None): self.n_confounder = Z
        else: assert self.n_confounder == Z, f"conf 列数不一致：CSV={Z}, config={self.n_confounder}"
        if not getattr(self, "n_cause", None): self.n_cause = T
        else: assert self.n_cause == T, f"cause 列数不一致：CSV={T}, config={self.n_cause}"
        if not getattr(self, "n_outcome", None): self.n_outcome = Y
        else: assert self.n_outcome == Y, f"outcome 列数不一致：CSV={Y}, config={self.n_outcome}"

        # 4) 按需下采样（用于 real_500/1000/1500 等）
        target = getattr(self, "sample_size", None)
        if not target or target <= 0 or target > N:
            target = N
        if target < N:
            rng = np.random.RandomState(42)  # 稳定复现
            idx = rng.choice(N, size=target, replace=False)
            conf, cause, out = conf[idx], cause[idx], out[idx]
            N = target

        # 5) 基础缓存（float32），全量
        self.confounder = conf.astype(np.float32)   # [N, Z]
        self.cause      = cause.astype(np.float32)  # [N, T]
        self.outcome    = out.astype(np.float32)    # [N, Y]
        self.sample_size = N

        # 6) 切分比例（默认 0.7/0.1/0.2，或用传入的 train_frac/val_frac）
        train_frac = getattr(self, "train_frac", 0.7)
        val_frac   = getattr(self, "val_frac",   0.1)
        if train_frac <= 0 or train_frac >= 1: train_frac = 0.7
        if val_frac   < 0 or (train_frac + val_frac) >= 1: val_frac = 0.1

        n_train = int(train_frac * N)
        n_val   = int(val_frac   * N)
        n_test  = N - n_train - n_val

        # 7) 切分并缓存（分别保存，供 generate_test_real 使用）
        self.confounder_train = self.confounder[:n_train]
        self.confounder_val   = self.confounder[n_train:n_train+n_val]
        self.confounder_test  = self.confounder[n_train+n_val:]

        self.cause_train = self.cause[:n_train]
        self.cause_val   = self.cause[n_train:n_train+n_val]
        self.cause_test  = self.cause[n_train+n_val:]

        self.outcome_train = self.outcome[:n_train]
        self.outcome_val   = self.outcome[n_train:n_train+n_val]
        self.outcome_test  = self.outcome[n_train+n_val:]

        # 8) 拼接 X 并构造 TensorDataset
        X_tr = np.concatenate([self.confounder_train, self.cause_train], axis=1)
        X_va = np.concatenate([self.confounder_val,   self.cause_val],   axis=1)
        X_te = np.concatenate([self.confounder_test,  self.cause_test],  axis=1)

        self.train_size, self.val_size, self.test_size = n_train, n_val, n_test
        self.outcome_list = [self.outcome]  # 保持与下游评估兼容

        import torch
        from torch.utils.data import TensorDataset
        self.train_dataset = TensorDataset(torch.tensor(X_tr), torch.tensor(self.outcome_train))
        self.valid_dataset = TensorDataset(torch.tensor(X_va), torch.tensor(self.outcome_val))
        self.x_test = torch.tensor(X_te)
        self.y_test = torch.tensor(self.outcome_test)

    def generate_real0(self):
        # load data
        cause = pds.read_csv("real_data/cause.csv").values
        confounder = pds.read_csv("real_data/confounder.csv").values

        assert cause.shape == (3080, 5)
        assert confounder.shape == (3080, 17)
        #
        # tv_sample_size = self.sample_size
        #
        # self.sample_size = confounder.shape[0]
        # self.train_size = int(tv_sample_size * 0.9)
        # self.val_size = int(tv_sample_size * 0.1) 

        outcomes = self.generate_real_outcome(cause, confounder, noise=0.1)
        self.cause = cause
        self.confounder = confounder
        self.outcome = outcomes
        return outcomes

    def generate(self):
        if not self.generated:
            if self.real_data:
                self.generate_real()
            else:
                self.get_coefficient_confounder_cause()
                self.get_coefficient_cause_cause()
                self.get_coefficient_outcome()

                self.generate_confounder()
                self.generate_cause()
                self.generate_outcome()
        self.generated = True
        return self.confounder, self.cause, self.outcome

    def split_xy(self, x, y):
        x_train = x[: self.train_size]
        y_train = y[: self.train_size]

        x_val = x[self.train_size : self.train_size + self.val_size]
        y_val = y[self.train_size : self.train_size + self.val_size]

        x_test = x[self.train_size + self.val_size :]
        y_test = y[self.train_size + self.val_size :]

        train_dataset = torch.utils.data.dataset.TensorDataset(self._make_tensor(x_train), self._make_tensor(y_train))
        valid_dataset = torch.utils.data.dataset.TensorDataset(self._make_tensor(x_val), self._make_tensor(y_val))
        x_test = self._make_tensor(x_test)
        y_test = self._make_tensor(y_test)
        return train_dataset, valid_dataset, x_test, y_test

    def generate_dataset(self):
        """
        real_data=True：直接调用 generate_real 并返回拆分后的数据集；
        否则：走原合成分支（保持你现有逻辑）。
        """
        import numpy as np
        import torch
        from torch.utils.data import TensorDataset

        if getattr(self, "real_data", False):
            if not hasattr(self, "train_dataset"):
                self.generate_real()
            return self.train_dataset, self.valid_dataset, self.x_test, self.y_test

        # ⬇️ 以下为合成分支（与你原先等价）
        self.generate()
        x = np.concatenate((self.confounder, self.cause), axis=-1)
        y = self.outcome
        N = x.shape[0]

        train_frac = getattr(self, "train_frac", 0.7)
        val_frac   = getattr(self, "val_frac",   0.1)
        if train_frac <= 0 or train_frac >= 1: train_frac = 0.7
        if val_frac   < 0 or (train_frac + val_frac) >= 1: val_frac = 0.1

        n_train = int(train_frac * N)
        n_val   = int(val_frac   * N)
        n_test  = N - n_train - n_val

        x_train, y_train = x[:n_train],                 y[:n_train]
        x_valid, y_valid = x[n_train:n_train+n_val],    y[n_train:n_train+n_val]
        x_test,  y_test  = x[n_train+n_val:],           y[n_train+n_val:]

        train_dataset = TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
        valid_dataset = TensorDataset(torch.tensor(x_valid), torch.tensor(y_valid))
        x_test_t = torch.tensor(x_test)
        y_test_t = torch.tensor(y_test)

        self.train_size, self.val_size, self.test_size = n_train, n_val, n_test
        self.sample_size = N

        return train_dataset, valid_dataset, x_test_t, y_test_t


    def generate_dataset0(self, return_dataset=True, weight=None):
        self.generate()

        if weight is None:
            x = np.concatenate((self.confounder, self.cause), axis=-1)
        else:
            x = np.concatenate((self.confounder, self.cause, weight), axis=-1)
        y = self.outcome

        if return_dataset:
            # x: n_confounder + K_cause + 1
            # y: n_cause - k_cause - 1
            return self.split_xy(x, y)
        else:
            return self._make_tensor(x), self._make_tensor(y)

    def generate_dataset_bmc(self, return_dataset=True, weight=None, npc=3):
        self.generate()

        pca = PCA(n_components=npc)
        self.pca = pca
        pc = pca.fit_transform(self.cause)

        if weight is None:
            x = np.concatenate((self.confounder, pc, self.cause), axis=-1)
        else:
            x = np.concatenate((self.confounder, pc, self.cause, weight), axis=-1)
        y = self.outcome

        if return_dataset:
            # x: n_confounder + K_cause + 1
            # y: n_cause - k_cause - 1
            return self.split_xy(x, y)
        else:
            return self._make_tensor(x), self._make_tensor(y)

    def generate_dataset_tarnet(self, return_dataset=True):
        self.generate()

        cause = self._make_tensor(self.cause)
        weight_vector = torch.pow(2, torch.arange(0, self.n_cause))
        cause_ind = torch.matmul(cause, weight_vector.to(cause)).unsqueeze(-1)

        x = torch.cat([self._make_tensor(self.confounder), cause_ind], dim=1)
        y = self._make_tensor(self.outcome)

        if return_dataset:
            # x: n_confounder + 1
            # y: 1
            return self.split_xy(x, y)
        else:
            return x, y

    def generate_dataset_propensity(self, return_dataset=True):
        self.generate()

        x = self._make_tensor(self.confounder)
        cause = self._make_tensor(self.cause)
        weight_vector = torch.pow(2, torch.arange(0, self.n_cause))
        y = torch.matmul(cause, weight_vector.to(cause)).squeeze().to(torch.long)

        if return_dataset:
            # x: n_confounder + K_cause + 1
            # y: n_cause - k_cause - 1
            return self.split_xy(x, y)
        else:
            return x, y

    def generate_dataset_vae(self, return_dataset=True):
        self.generate()
        x = self._make_tensor(self.cause)
        y = x.detach().clone()

        if return_dataset:
            # x: K_cause
            # y: k_cause
            return self.split_xy(x, y)
        else:
            return x, y

    def generate_dataset_dr(self, z, z_rand, return_dataset=True, shuffle=True):
        self.generate()

        x1 = torch.cat([self._make_tensor(self.confounder), z], dim=-1)
        x0 = torch.cat([self._make_tensor(self.confounder), z_rand], dim=-1)
        x = torch.cat([x1, x0], dim=0)
        y = torch.cat([torch.ones(x1.shape[0]), torch.zeros(x1.shape[0])]).unsqueeze(-1).to(x1)

        if shuffle:
            ind = torch.randperm(x.shape[0])
            x = x[ind]
            y = y[ind]

        if return_dataset:
            # x: K_cause
            # y: k_cause
            return self.split_xy(x, y)
        else:
            return x, y

    def generate_dataset_potential_cause(self, k_cause, new_cause=None, return_dataset=True, predict_all_causes=False):
        # k_cause in [0, n_cause - 1)
        # assert k_cause != self.n_cause - 1
        self.generate()

        if new_cause is None:
            cause = self.cause
        else:
            cause = new_cause

        single_cause = cause[:, k_cause : (k_cause + 1)]

        ancestor_cause = cause[:, :k_cause]
        try:
            descendent_cause = cause[:, self.descendant[k_cause]]
        except TypeError:
            return None

        if descendent_cause.shape[1] == 0:
            return None

        if ancestor_cause.shape == 2:
            x = np.concatenate((self.confounder, ancestor_cause, single_cause), axis=-1)
        else:
            x = np.concatenate((self.confounder, single_cause), axis=-1)

        y = descendent_cause

        if predict_all_causes:
            # override all previous settings
            x = np.concatenate((self.confounder, single_cause), axis=-1)
            y = np.delete(cause, k_cause, axis=1)

        if return_dataset:
            # x: n_confounder + K_cause + 1
            # y: n_cause - k_cause - 1
            return self.split_xy(x, y)
        else:
            return self._make_tensor(x), self._make_tensor(y)

    #
    #
    # def get_one_cause(self, i, confounder_factor_mean, previous_causes):
    #     # K
    #     mean = confounder_factor_mean[:, i]
    #
    #     if i > 0:
    #         cause_factor_mean = np.matmul(previous_causes, self.coefficient_cause_cause[:i, i]) / previous_causes.shape[1]
    #     else:
    #         cause_factor_mean = 0
    #     cause_error = np.random.randn(self.sample_size) * self.cause_noise
    #     cause_logit = mean + cause_factor_mean + cause_error
    #     cause = np.random.binomial(1, logistic(cause_logit))
    #     return cause, cause_logit, cause_error
    #
    # def generate_cause(self):
    #     confounder_factor_mean = np.matmul(self.confounder, self.coefficient_confounder_cause) / self.n_confounder * self.confounding_level
    #
    #     cause_list = []
    #     cause_logit_list = []
    #     cause_error_list = []
    #     for i in range(self.n_cause):
    #         if i > 0:
    #             previous_causes = np.stack(cause_list, axis=-1)
    #         else:
    #             previous_causes = None
    #
    #         new_cause, cause_logit, cause_error = self.get_one_cause(i, confounder_factor_mean, previous_causes)
    #         cause_list.append(new_cause)
    #         cause_logit_list.append(cause_logit)
    #         cause_error_list.append(cause_error)
    #
    #     self.cause = np.stack(cause_list, axis=-1)
    #     self.cause_logit = np.stack(cause_logit_list, axis=-1)
    #     self.cause_error = np.stack(cause_error_list, axis=-1)
    #
    #
    #
    def get_x_potential_cause_oracle(self, k_cause, return_cause=False):
        print("Use Oracle")
        confounder_factor_mean = (
            np.matmul(self.confounder, self.coefficient_confounder_cause) / self.n_confounder * self.confounding_level
        )

        cause_list = []
        for i in range(self.n_cause):

            if i < k_cause:
                cause_list.append(self.cause[:, i])
            elif i == k_cause:
                cause_list.append(1.0 - self.cause[:, i])
            else:
                previous_causes = np.stack(cause_list, axis=-1)
                mean = confounder_factor_mean[:, i]
                cause_factor_mean = (
                    np.matmul(previous_causes, self.coefficient_cause_cause[:i, i]) / previous_causes.shape[1]
                )
                cause_error = self.cause_error[:, i]
                cause_logit = mean + cause_factor_mean + cause_error
                new_cause = np.random.binomial(1, logistic(cause_logit))
                cause_list.append(new_cause)
        cause = np.stack(cause_list, axis=-1)
        x = np.concatenate((self.confounder, cause), axis=-1)
        if return_cause:
            return cause, self._make_tensor(x)
        else:
            return self._make_tensor(x)

    def get_x_potential_cause(self, k_cause, potential_cause, predict_all_causes=False):
        if not predict_all_causes:
            potential_cause = potential_cause.cpu().numpy()
            cause = self.cause.copy()

            cause[:, k_cause] = 1.0 - cause[:, k_cause]

            for i, j in enumerate(self.descendant[k_cause]):
                cause[:, j] = potential_cause[:, i]
        else:
            potential_cause = potential_cause.cpu().numpy()
            cause = self.cause.copy()

            for i in range(cause.shape[1]):
                if i == k_cause:
                    cause[:, k_cause] = 1.0 - cause[:, k_cause]
                elif i < k_cause:
                    cause[:, i] = potential_cause[:, i]
                else:
                    cause[:, i] = potential_cause[:, i - 1]

        x = np.concatenate((self.confounder, cause), axis=-1)
        return self._make_tensor(x)

    def generate_counterfactual_test(self, n_flip, weight=None):

        new_cause = self.flip_cause(n_flip)
        new_outcome = self.generate_counterfactual(new_cause)
        new_x = np.concatenate((self.confounder, new_cause), axis=-1)

        cate_test = self._make_tensor((new_outcome - self.outcome)[self.train_size + self.val_size :])
        new_x_test = self._make_tensor(new_x[self.train_size + self.val_size :])
        if weight is not None:
            new_x_test = torch.cat([new_x_test, new_x_test[:, 0:1]], dim=-1)

        # todo: generate all counterfactual outcomes
        outcome_list = []
        for x in range(self.n_treatment):
            # 1, K
            cause_setting = self.cause_ref[x : x + 1, :]
            new_cause = np.concatenate([cause_setting] * self.sample_size, axis=0)

            this_outcome = self.generate_counterfactual(new_cause)

            assert this_outcome.shape[0] == self.cause.shape[0]
            assert this_outcome.shape[1] == 1
            outcome_list.append(this_outcome)
        self.outcome_list = outcome_list
        return new_x_test, cate_test

    def generate_counterfactual_test_bmc(self, n_flip, weight=None):

        new_cause = self.flip_cause(n_flip)
        pc = self.pca.transform(new_cause)
        new_outcome = self.generate_counterfactual(new_cause)
        new_x = np.concatenate((self.confounder, pc, new_cause), axis=-1)

        cate_test = self._make_tensor((new_outcome - self.outcome)[self.train_size + self.val_size :])
        new_x_test = self._make_tensor(new_x[self.train_size + self.val_size :])
        if weight is not None:
            new_x_test = torch.cat([new_x_test, new_x_test[:, 0:1]], dim=-1)

        # todo: generate all counterfactual outcomes
        outcome_list = []
        for x in range(self.n_treatment):
            # 1, K
            cause_setting = self.cause_ref[x : x + 1, :]
            new_cause = np.concatenate([cause_setting] * self.sample_size, axis=0)

            this_outcome = self.generate_counterfactual(new_cause)

            assert this_outcome.shape[0] == self.cause.shape[0]
            assert this_outcome.shape[1] == 1
            outcome_list.append(this_outcome)
        self.outcome_list = outcome_list
        return new_x_test, cate_test
    def generate_test_real(self):
        """
        为真实数据生成一组“测试集单因子干预”的 X：
        对测试集的每一列 treatment（cause 的列）逐列翻转（0->1, 1->0），
        返回一个列表 [X_flip_col0, X_flip_col1, ..., X_flip_col{T-1}]，
        其中每个元素形状都是 [n_test, Z+T]，与 self.x_test 对齐。
        """
        import numpy as np

        assert hasattr(self, "confounder_test") and hasattr(self, "cause_test"), \
            "请先调用 generate_real() 以构造测试集"

        Z = self.confounder_test.shape[1]
        T = self.cause_test.shape[1]
        n_test = self.confounder_test.shape[0]
        assert self.cause_test.shape[0] == n_test, "测试集行数不一致"

        new_x_list = []
        for t_idx in range(T):
            new_cause = self.cause_test.copy()
            new_cause[:, t_idx] = 1.0 - new_cause[:, t_idx]  # 单列翻转
            new_x = np.concatenate([self.confounder_test, new_cause], axis=1)
            new_x_list.append(new_x.astype(np.float32))

        return new_x_list

    def generate_test_real0(self, weight=None):
        new_x_list = []

        n_treatment = int(np.power(2, self.n_cause))

        for x in range(n_treatment):
            # 1, K
            cause_setting = self.cause_ref[x : x + 1, :]
            # N, K
            new_cause = np.concatenate([cause_setting] * self.confounder.shape[0], axis=0)
            # N, K + D + 1
            new_x = np.concatenate((self.confounder, new_cause), axis=-1)

            new_x_test = self._make_tensor(new_x[self.train_size + self.val_size :])

            if weight is not None:
                new_x_test = torch.cat([new_x_test, new_x_test[:, 0:1]], dim=-1)

            new_x_list.append(new_x_test)
        return new_x_list

    def generate_test_real_bmc(self, weight=None):
        new_x_list = []

        n_treatment = int(np.power(2, self.n_cause))

        for x in range(n_treatment):
            # 1, K
            cause_setting = self.cause_ref[x : x + 1, :]
            # N, K
            new_cause = np.concatenate([cause_setting] * self.confounder.shape[0], axis=0)

            pc = self.pca.transform(new_cause)
            # N, K + D + 1
            new_x = np.concatenate((self.confounder, pc, new_cause), axis=-1)

            new_x_test = self._make_tensor(new_x[self.train_size + self.val_size :])

            if weight is not None:
                new_x_test = torch.cat([new_x_test, new_x_test[:, 0:1]], dim=-1)

            new_x_list.append(new_x_test)
        return new_x_list

    def generate_test_tarnet(self, weight=None):
        new_x_list = []

        n_treatment = int(np.power(2, self.n_cause))

        for x in range(n_treatment):
            # 1, K
            cause_setting = self.cause_ref[x : x + 1, :]
            # N, K
            new_cause = np.concatenate([cause_setting] * self.confounder.shape[0], axis=0)

            cause_ind = cause_to_num(new_cause)[:, None]
            # N, K + D + 1
            new_x = np.concatenate((self.confounder, cause_ind), axis=-1)

            new_x_test = self._make_tensor(new_x[self.train_size + self.val_size :])

            if weight is not None:
                new_x_test = torch.cat([new_x_test, new_x_test[:, 0:1]], dim=-1)

            new_x_list.append(new_x_test)
        return new_x_list
   
    def evaluate_real(self, y_list, threshold=0.5, save_path=None):
        """
        真实数据：通常没有个体级真值 tau，无法计算 PEHE。
        本函数：
        - 如果存在非空的 self.tau（模拟数据），则计算 PEHE（保持向后兼容）
        - 否则（真实数据），只做预测结果的统计摘要，不再尝试计算 PEHE

        Args:
            y_list: list[np.ndarray]，每个元素是某个干预场景下的预测结果，形状 (N, 1) 或 (N,)
            threshold: 若预测为概率，按该阈值给出阳性率（默认 0.5）
            save_path: 若给定字符串路径，则将摘要保存为 CSV
        """
        import numpy as _np
        import pandas as _pd

        # ---- 1) 统一整理形状，拼成 (N, K) 的矩阵 ----
        if y_list is None or len(y_list) == 0:
            print("[evaluate_real] y_list 为空，跳过评估。")
            return None

        cols = []
        for idx, y in enumerate(y_list):
            if y is None:
                continue
            y = _np.asarray(y)
            if y.ndim == 0:
                # 单个标量，跳过
                continue
            if y.ndim == 1:
                y = y.reshape(-1, 1)
            elif y.ndim >= 2:
                # 只保留第一列（通常 n_outcome=1）
                y = y.reshape(y.shape[0], -1)[:, :1]
            cols.append(y)

        if len(cols) == 0:
            print("[evaluate_real] y_list 内无有效数组，跳过评估。")
            return None

        y_mat = _np.concatenate(cols, axis=1)  # (N, K)
        N, K = y_mat.shape
        print(f"[evaluate_real] 收到预测矩阵形状: {y_mat.shape} (N×K)")

        # ---- 2) 如果存在模拟真值 tau（并且非空），仍然计算 PEHE（向后兼容）----
        tau = getattr(self, "tau", None)
        if tau is not None:
            tau = _np.asarray(tau)
            # 允许 tau 是 (N, K_true) 或 (N,) 或 (N,1)。若 K 不匹配就不算 PEHE。
            if tau.size > 0:
                if tau.ndim == 1:
                    tau = tau.reshape(-1, 1)
                elif tau.ndim >= 2:
                    tau = tau.reshape(tau.shape[0], -1)
                if tau.shape[0] == N and tau.shape[1] == K:
                    pehe = _np.sqrt(_np.mean((tau - y_mat) ** 2))
                    print(f"[evaluate_real] 检测到模拟真值，PEHE = {pehe:.6f}")
                    return {"mode": "synthetic_with_tau", "PEHE": float(pehe)}
                else:
                    print(
                        "[evaluate_real] 检测到 tau，但形状不匹配："
                        f"tau={tau.shape}, y_mat={y_mat.shape}；跳过 PEHE。"
                    )

        # ---- 3) 真实数据：输出统计摘要（不计算 PEHE）----
        # 统计：均值、标准差、分位数、若似概率则阳性率
        means = _np.mean(y_mat, axis=0)
        stds = _np.std(y_mat, axis=0)
        q05 = _np.quantile(y_mat, 0.05, axis=0)
        q25 = _np.quantile(y_mat, 0.25, axis=0)
        q50 = _np.quantile(y_mat, 0.50, axis=0)
        q75 = _np.quantile(y_mat, 0.75, axis=0)
        q95 = _np.quantile(y_mat, 0.95, axis=0)

        # 判定“像概率”：所有值均在[0,1]内
        looks_prob = _np.isfinite(y_mat).all() and (y_mat >= 0.0).all() and (y_mat <= 1.0).all()
        if looks_prob:
            pos_rate = _np.mean(y_mat > threshold, axis=0)
        else:
            pos_rate = _np.full(K, _np.nan)

        df = _pd.DataFrame({
            "scenario_id": _np.arange(K),
            "mean": means,
            "std": stds,
            "q05": q05, "q25": q25, "q50": q50, "q75": q75, "q95": q95,
            "positive_rate" if looks_prob else "positive_rate(NaN)": pos_rate,
        })

        # 打印一个简表
        with _pd.option_context("display.max_columns", None, "display.width", 120):
            print("[evaluate_real] 真实数据：各干预场景预测摘要（前几行）：")
            print(df.head(min(10, K)))

        # 可选保存
        if save_path:
            try:
                df.to_csv(save_path, index=False)
                print(f"[evaluate_real] 摘要已保存到: {save_path}")
            except Exception as e:
                print(f"[evaluate_real] 保存 CSV 失败: {e}")

        return {
            "mode": "real_no_tau",
            "summary": df,
            "looks_prob": bool(looks_prob),
            "threshold": float(threshold) if looks_prob else None,
            "shape": (int(N), int(K)),
        }

    def evaluate_real0(self, y_list):

        y_mat_true = np.concatenate(self.outcome_list, axis=-1)

        # N, 2^K
        y_mat = np.concatenate(y_list, axis=-1)

        # PEHE
        tau_hat = y_mat[:, 1:] - y_mat[:, :-1]
        n_test = tau_hat.shape[0]
        tau = y_mat_true[-n_test:, 1:] - y_mat_true[-n_test:, :-1]

        pehe = np.sqrt(np.mean((tau - tau_hat) ** 2))
        pehe_sd = bootstrap_RMSE((torch.tensor((tau - tau_hat).flatten()) ** 2))  # pylint: disable=not-callable

        rmse = np.sqrt(np.mean((y_mat_true[-n_test:, :] - y_mat) ** 2))
        rmse_sd = bootstrap_RMSE(
            (torch.tensor((y_mat_true[-n_test:, :] - y_mat).flatten()) ** 2)  # pylint: disable=not-callable
        )

        # NDCG
        order = y_mat.argsort(axis=-1)
        # order2 = order.copy()
        # N, 2^K
        ranks_predicted = order.argsort(axis=-1)
        rel = self.get_relevance_score(ranks_predicted)
        scores = ndcg_at_k(rel, 5)
        mean_ndcg = np.mean(scores)
        sd_ndcg = np.std(scores) / np.sqrt(scores.shape[0])

        # ranking dist
        rank_dist = np.sum(np.abs(self.ranks[self.val_size + self.train_size :, :] - ranks_predicted), axis=-1)
        rank_mean = np.mean(rank_dist)
        rank_sd = np.std(rank_dist) / np.sqrt(rank_dist.shape[0])

        # improvements
        order2 = np.argmin(y_mat, axis=-1)[:, None]
        improvements = self.get_predicted_improvements(order2, k=1)
        mean_improvements = np.median(improvements)
        sd_improvements = np.std(improvements) / np.sqrt(improvements.shape[0])

        oracle = self.generate_oracle_improvements()
        mean_oracle = np.median(oracle)
        sd_oracle = np.std(oracle) / np.sqrt(oracle.shape[0])

        print(
            round(float(mean_ndcg), 3),
            round(sd_ndcg, 3),
            round(float(mean_improvements), 3),
            round(sd_improvements, 3),
            round(float(rank_mean), 3),
            round(rank_sd, 3),
            round(float(pehe), 3),
            round(pehe_sd, 3),
            round(float(rmse), 3),
            round(rmse_sd, 3),
        )
        # print(round(float(mean_ndcg), 3), round(sd_ndcg, 3))
        # print(round(float(mean_improvements), 3), round(sd_improvements, 3))
        # print(round(float(mean_oracle), 3), round(sd_oracle, 3))
        # print(round(float(rank_mean), 3), round(rank_sd, 3))

    def generate_counterfactual_test_tarnet(self, n_flip, weight=None):

        new_cause = self._make_tensor(self.flip_cause(n_flip))

        weight_vector = torch.pow(2, torch.arange(0, self.n_cause))
        cause_ind = torch.matmul(new_cause, weight_vector.to(new_cause)).unsqueeze(-1)

        x = torch.cat([self._make_tensor(self.confounder), cause_ind], dim=1)

        new_outcome = self.generate_counterfactual(new_cause.cpu().numpy())

        cate_test = self._make_tensor((new_outcome - self.outcome)[self.train_size + self.val_size :])
        new_x_test = self._make_tensor(x[self.train_size + self.val_size :])
        if weight is not None:
            new_x_test = torch.cat([new_x_test, new_x_test[:, 0:1]], dim=-1)

        # generate all counterfactual outcomes
        outcome_list = []
        for x in range(self.n_treatment):
            # 1, K
            cause_setting = self.cause_ref[x : x + 1, :]
            new_cause = np.concatenate([cause_setting] * self.confounder.shape[0], axis=0)

            this_outcome = self.generate_counterfactual(new_cause)

            assert this_outcome.shape[0] == self.cause.shape[0]
            assert this_outcome.shape[1] == 1
            outcome_list.append(this_outcome)
        self.outcome_list = outcome_list
        return new_x_test, cate_test

    def flip_cause(self, n_flip):
        flip_index = [np.random.choice(self.n_cause, n_flip, False) for x in range(self.sample_size)]
        flip_index = np.stack(flip_index, axis=0)
        # print('flip_index', flip_index)
        flip_onehot = np.zeros((self.sample_size, self.n_cause))
        for i in range(flip_index.shape[1]):
            tmp = np.zeros((self.sample_size, self.n_cause))
            tmp[np.arange(self.sample_size), flip_index[:, i]] = 1
            flip_onehot += tmp
        new_cause = self.cause * (1 - flip_onehot) + (1.0 - self.cause) * flip_onehot
        return new_cause


if __name__ == "__main__":
    dg = DataGenerator(
        n_confounder=1,
        n_cause=2,
        n_outcome=1,
        sample_size=4,
        p_confounder_cause=0,
        p_cause_cause=1,
        cause_noise=0,
        outcome_noise=0,
        linear=True,
    )
    confounder1, cause1, outcome1 = dg.generate()
    # print(confounder)
    print(cause1)

    # print(dg.coefficient_confounder_outcome)
    # print(dg.coefficient_cause_outcome)
    # print(dg.coefficient_cause_cause)
    new_cause1 = dg.flip_cause(2)
    print(new_cause1)
    print(outcome1)
    print(dg.generate_counterfactual(new_cause1))
