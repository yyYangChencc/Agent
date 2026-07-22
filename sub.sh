#!/bin/bash
# 作业名称。
#SBATCH --job-name=agent-flan-xl
# 使用 A800 GPU 队列。
#SBATCH --partition=A800-N
# 使用一个节点和一个任务。
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# 每个任务使用一个 CPU 核心。
#SBATCH --cpus-per-task=1
# 使用一张 GPU。
#SBATCH --gres=gpu:1
# 将标准输出和错误分别写入带作业编号的日志。
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

# 前三个参数必须由提交者提供，避免绑定特定集群账号目录。
CONDA_ENV_PATH="/data/user/tianzhiliang/yyc/venv"
FLAN_MODEL_DIR="/data/user/tianzhiliang/public/models/google_flan-t5-xl"
SCENARIO="iac_gay_marriage_30_same_side_isolated"
TICKS="100"
SEED="42"

# 可通过 sbatch --export 覆盖以下实验模式。
VERSION="${VERSION:-full}"
OPINION_MODE="${OPINION_MODE:-llm_as_judge}"
PSYCHOLOGY_MODE="${PSYCHOLOGY_MODE:-llm}"
LLM_MODE="${LLM_MODE:-real}"

# 加载集群提供的 CUDA 和 Anaconda 环境。
module load cuda/11.8
module load anaconda3
source activate "$CONDA_ENV_PATH"

# 必须从项目根目录执行 sbatch sub.sh，SLURM 会记录该提交目录。
PROJECT_DIR="${SLURM_SUBMIT_DIR:?必须通过 sbatch 从项目根目录提交作业}"
RUN_SCRIPT="$PROJECT_DIR/backend/run_experiment.py"

if [ ! -f "$RUN_SCRIPT" ]; then
    echo "未找到实验入口: $RUN_SCRIPT" >&2
    exit 1
fi

# 在启动作业前检查 Transformers 加载所需的关键模型文件。
if [ ! -f "$FLAN_MODEL_DIR/config.json" ]; then
    echo "FLAN 模型目录缺少 config.json: $FLAN_MODEL_DIR" >&2
    exit 1
fi
if [ ! -f "$FLAN_MODEL_DIR/model.safetensors.index.json" ]; then
    echo "FLAN 模型目录缺少 model.safetensors.index.json: $FLAN_MODEL_DIR" >&2
    exit 1
fi

cd "$PROJECT_DIR/backend"
export PYTHONUNBUFFERED=1

# 直接运行后端实验，不启动前端或 Web 服务。
python -u run_experiment.py \
    --scenario "$SCENARIO" \
    --ticks "$TICKS" \
    --seed "$SEED" \
    --version "$VERSION" \
    --opinion-mode "$OPINION_MODE" \
    --psychology-mode "$PSYCHOLOGY_MODE" \
    --llm "$LLM_MODE" \
    --flan-model "$FLAN_MODEL_DIR"
