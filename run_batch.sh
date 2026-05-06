#!/bin/bash
# ============================================================
# 通用批量编辑实验启动脚本
# 支持 MEMIT / PMET，cf / zsre，多种模型
# ============================================================

set -e

# -------------------- 参数解析 --------------------
ALG=${1:-}     # MEMIT | PMET
MODEL=${2:-}   # llama3 | gptj | gptneox
DS=${3:-}      # cf | zsre
NUM=${4:-2000} # 编辑数量，默认 2000

# -------------------- 帮助信息 --------------------
show_help() {
    cat << 'EOF'
用法: bash run_batch.sh <算法> <模型> <数据集> [编辑数]

参数说明:
  算法    MEMIT  - 多跳编辑（推荐）
           PMET  - 精确记忆编辑
  模型    llama3  - LLaMA-3-8B-Instruct
           gptj   - GPT-J-6B
           gptneox- GPT-NeoX-20B
  数据集  cf    - CounterFact
           zsre  - ZsRe
  编辑数  整数，默认 2000

示例:
  bash run_batch.sh MEMIT llama3 zsre 2000
  bash run_batch.sh PMET  llama3 cf    500
  bash run_batch.sh MEMIT gptj  zsre  1000
EOF
}

# -------------------- 参数校验 --------------------
if [[ -z "$ALG" || -z "$MODEL" || -z "$DS" ]]; then
    echo "错误: 必须提供算法、模型和数据集"
    show_help
    exit 1
fi

if [[ "$ALG" != "MEMIT" && "$ALG" != "PMET" ]]; then
    echo "错误: 算法必须是 MEMIT 或 PMET"
    exit 1
fi

if [[ "$DS" != "cf" && "$DS" != "zsre" ]]; then
    echo "错误: 数据集必须是 cf 或 zsre"
    exit 1
fi

# -------------------- 模型映射 --------------------
case "$MODEL" in
    llama3)
        MODEL_NAME_CHOICE="my_llama-3-8b-instruct"
        HPARAMS_FNAME="meta-llama_Meta-Llama-3-8B-Instruct.json"
        if [[ "$ALG" == "MEMIT" ]]; then
            MODEL_PATH="/home/wentao/xzw1/LLM/my_llama-3-8b-instruct"
        else
            MODEL_PATH="/home/wentao/xzw1/LLM/Meta-Llama-3-8B-Instruct"
        fi
        ;;
    gptj)
        MODEL_NAME_CHOICE="EleutherAI/gpt-j-6B"
        HPARAMS_FNAME="EleutherAI_gpt-j-6B.json"
        MODEL_PATH="/home/wentao/xzw1/LLM/gpt-j-6B"
        ;;
    gptneox)
        MODEL_NAME_CHOICE="EleutherAI/gpt-neox-20b"
        HPARAMS_FNAME="EleutherAI_gpt-neox-20b.json"
        MODEL_PATH="/home/wentao/xzw1/LLM/gpt-neox-20b"
        ;;
    *)
        echo "错误: 不支持的模型 $MODEL"
        echo "支持的模型: llama3, gptj, gptneox"
        exit 1
        ;;
esac

# -------------------- CUDA 设备分配 --------------------
# MEMIT 用 2 号卡，PMET 用 3 号卡
if [[ "$ALG" == "MEMIT" ]]; then
    CUDA_DEV=2
    CONDA_ENV="memit_dqg"
    RUN_DIR="~/xzw1/dqg/memit"
else
    CUDA_DEV=3
    CONDA_ENV="pmet_dqg"
    RUN_DIR="~/xzw1/dqg/PMET/edit"
fi

# -------------------- 构建命令 --------------------
echo "=========================================="
echo "  算法:    $ALG"
echo "  模型:    $MODEL ($MODEL_NAME_CHOICE)"
echo "  数据集:  $DS"
echo "  编辑数:  $NUM"
echo "  CUDA:    $CUDA_DEV"
echo "  Conda:   $CONDA_ENV"
echo "=========================================="

cd "$RUN_DIR"

# 取消 SSL 证书限制（避免 HuggingFace 下载问题）
unset SSL_CERT_FILE
unset CURL_CA_BUNDLE

export CUDA_VISIBLE_DEVICES=$CUDA_DEV

if [[ "$ALG" == "MEMIT" ]]; then
    # MEMIT: --model_name 传路径，--hparams_fname 只写文件名
    python -m experiments.evaluate \
        --alg_name MEMIT \
        --model_name "$MODEL_PATH" \
        --hparams_fname "$HPARAMS_FNAME" \
        --num_edits "$NUM" \
        --ds_name "$DS" \
        --skip_generation_tests
else
    # PMET: --model_name 传 choices 字符串，--model_path 传实际路径
    python evaluate.py \
        --alg_name PMET \
        --model_name "$MODEL_NAME_CHOICE" \
        --model_path "$MODEL_PATH" \
        --hparams_fname "$HPARAMS_FNAME" \
        --num_edits "$NUM" \
        --ds_name "$DS" \
        --skip_generation_tests
fi
