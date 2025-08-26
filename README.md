# Estimating Multi-cause Treatment Effects via Single-cause Perturbation (NeurIPS 2021)

**Single-cause Perturbation (SCP)** is a framework for estimating the *multi-cause conditional average treatment effect (CATE)* from observational data.

Most existing CATE methods assume *single-cause* interventions, i.e., only one variable can be manipulated at a time. However, many real-world applications involve interventions on *multiple causes simultaneously*. SCP addresses this challenge by:
- Leveraging the link between single- and multi-cause interventions.
- Using **data augmentation** to mitigate confounding bias.
- Avoiding strong assumptions on the distribution or functional form of the data-generating process (DGP).

---

## Installation

```bash
conda create -n scp38 python=3.8 -y
conda activate scp38
pip install -r requirement_new.txt
```

---

## Data Preparation

Place the real datasets under `real_data/`:
- `cause.csv`
- `confounder.csv`
- `outcome.csv`

---

## Running Experiments

Start with smaller samples and then scale up to the full dataset:

```bash
python -u -m run_simulation --method=scp --config=real_500   | tee results/scp_real_500.txt
python -u -m run_simulation --method=scp --config=real_1000  | tee results/scp_real_1000.txt
python -u -m run_simulation --method=scp --config=real_1500  | tee results/scp_real_1500.txt
python -u -m run_simulation --method=scp --config=real_full  | tee results/scp_real_full.txt
```

An implementation of SCP is provided in `run_simulation_scp.py`.  
In this implementation, we use **DR-CFR** in step one and **neural network regression** in step two.

