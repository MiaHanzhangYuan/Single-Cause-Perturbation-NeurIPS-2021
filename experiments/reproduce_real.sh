#!/usr/bin/env bash
set -euo pipefail

# 进入 repo 根目录
cd "$(dirname "$0")/.."

# 统一建齐目录（和 run_one_real.sh/summarise_real.sh 保持一致）
mkdir -p model results2021 logs

# 让 python 能从项目根导入包
export PYTHONPATH="$PWD"

echo -e "\n=============== Running: run_one_real.sh ===============\n"
# 同时落盘和打印
bash experiments/run_one_real.sh 2>&1 | tee logs/run_one_real.log

echo -e "\n=============== Running: summarise_real.sh ===============\n"
# 追加日志，不覆盖
bash experiments/summarise_real.sh 2>&1 | tee -a logs/summarise_real.log

echo -e "\nDone. Summaries should be under results2021/ and CSV under project root (e.g., summary_real.csv)."
