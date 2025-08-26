#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
汇总 real_500 / real_1000 / real_1500 等规模的实验结果到一张表
默认从 results2021/ 目录读入，输出 summary_real.csv
"""

import argparse
import glob
import os
import re
import pandas as pd

FLOAT_RE = re.compile(r"[-+]?(?:\d+\.\d+|\d+\.|\.\d+|\d+)(?:[eE][-+]?\d+)?")

def parse_scp_last_row(text: str):
    """
    解析 SCP 打印的 DataFrame 最后一行：
    形如：
    8  0.033384  0.031353 -0.008515  0.007452  0.028873  0.050537  0.082078  NaN
    返回 dict: {'mean':..., 'std':..., 'q05':..., 'q25':..., 'q50':..., 'q75':..., 'q95':...}
    """
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    # 只在最后 50 行里找，越靠后越可能是结果
    for ln in reversed(lines[-50:]):
        nums = FLOAT_RE.findall(ln)
        # 这一行通常会有 8-9 段，其中第一个是 scenario_id（整数或浮点），后面 7 个是统计量
        if len(nums) >= 8:
            try:
                # 丢弃第一个（scenario_id），取后面 7 个统计值
                vals = list(map(float, nums[1:8]))
                return {
                    "mean": vals[0],
                    "std": vals[1],
                    "q05": vals[2],
                    "q25": vals[3],
                    "q50": vals[4],
                    "q75": vals[5],
                    "q95": vals[6],
                }
            except Exception:
                pass
    return {}

def parse_generic_last_numbers(text: str):
    """
    兜底：从末尾向前找“至少3个浮点数”的一行，返回前 7 个（不够就缺失）
    """
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    for ln in reversed(lines[-50:]):
        nums = FLOAT_RE.findall(ln)
        if len(nums) >= 3:
            vals = [float(x) for x in nums[:7]]
            keys = ["m1","m2","m3","m4","m5","m6","m7"]
            return dict(zip(keys, vals))
    return {}

def infer_method_and_size(path: str):
    """
    从文件名推断方法名和样本量：
    例如 results2021/scp_real_500.txt -> ('scp', 500)
    """
    name = os.path.basename(path)
    m = re.match(r"([A-Za-z\-]+)_real_(\d+)\.txt$", name)
    if not m:
        return None, None
    method = m.group(1).lower()
    size = int(m.group(2))
    return method, size

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results2021", help="结果文件目录")
    ap.add_argument("--out", default="summary_real.csv", help="输出 CSV 路径")
    args = ap.parse_args()

    rows = []
    files = sorted(glob.glob(os.path.join(args.results_dir, "*real_*.txt")))
    if not files:
        print(f"[WARN] 没在 {args.results_dir} 里找到 *real_*.txt 文件")
    for fp in files:
        method, size = infer_method_and_size(fp)
        if method is None:
            print(f"[SKIP] 无法识别文件名: {fp}")
            continue
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read()

        parsed = {}
        status = "ok"

        if method in ("scp", "dor", "propensity", "vsr", "tarnet", "drcrn", "bmc", "overlap"):
            # 先尝试 SCP 风格的 DataFrame 末行
            parsed = parse_scp_last_row(txt)
            if not parsed:
                # 再尝试泛化解析
                parsed = parse_generic_last_numbers(txt)

        # 标注一些常见“空跑/占位”的情况
        if re.search(r"No such file or directory", txt, flags=re.I):
            status = "no_model_dir"
        if re.search(r"^\s*0\s*$", txt.strip()):
            status = "zero_output"

        row = {"method": method, "sample_size": size, "status": status}
        row.update(parsed)
        rows.append(row)

    if not rows:
        print("[WARN] 没有可写入的数据行")
        return

    df = pd.DataFrame(rows).sort_values(["method", "sample_size"]).reset_index(drop=True)

    # 把方法名规范化一点（首字母大写）
    df["method"] = df["method"].str.upper().str.replace("-", "")
    # 列顺序
    cols = ["method","sample_size","status","mean","std","q05","q25","q50","q75","q95","m1","m2","m3","m4","m5","m6","m7"]
    cols = [c for c in cols if c in df.columns]
    df = df[cols]

    df.to_csv(args.out, index=False, encoding="utf-8")
    print(f"[OK] 已生成汇总：{args.out}")
    try:
        # 额外输出一个“方法 × 样本量 → mean”的透视表，便于快速看趋势
        if "mean" in df.columns:
            pivot = df.pivot_table(index="method", columns="sample_size", values="mean", aggfunc="first")
            print("\n==== mean 透视表（方法 × 样本量）====")
            print(pivot)
    except Exception:
        pass

if __name__ == "__main__":
    main()
