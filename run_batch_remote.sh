#!/bin/bash
# ============================================================
# 一键启动：本地推送 Git → 服务器执行
# ============================================================
#
# 用法:
#   bash run_batch_remote.sh MEMIT llama3 zsre 2000
#   bash run_batch_remote.sh PMET  llama3 zsre 2000
#   bash run_batch_remote.sh MEMIT gptj  zsre 1000
#   bash run_batch_remote.sh MEMIT llama3 cf    500
#
# 参数: <算法> <模型> <数据集> [编辑数]
#   算法:   MEMIT | PMET
#   模型:   llama3 | gptj | gptneox
#   数据集: cf | zsre
#   编辑数: 默认 2000

set -e

ALG=${1:-}
MODEL=${2:-}
DS=${3:-}
NUM=${4:-2000}

# ------------------- 帮助信息 -------------------
if [[ -z "$ALG" || "$ALG" == "--help" || "$ALG" == "-h" ]]; then
    cat << 'EOF'
一键批量编辑实验启动器

用法:
  bash run_batch_remote.sh <算法> <模型> <数据集> [编辑数]

参数:
  算法    MEMIT  - 多跳编辑
           PMET  - 精确记忆编辑
  模型    llama3   - LLaMA-3-8B-Instruct
           gptj    - GPT-J-6B
           gptneox - GPT-NeoX-20B
  数据集  cf    - CounterFact
           zsre  - ZsRe
  编辑数  整数，默认 2000

快速示例:
  bash run_batch_remote.sh MEMIT llama3 zsre 2000
  bash run_batch_remote.sh PMET  llama3 cf    500
EOF
    exit 0
fi

# ------------------- 参数校验 -------------------
if [[ "$ALG" != "MEMIT" && "$ALG" != "PMET" ]]; then
    echo "[错误] 算法必须是 MEMIT 或 PMET"
    exit 1
fi
if [[ "$DS" != "cf" && "$DS" != "zsre" ]]; then
    echo "[错误] 数据集必须是 cf 或 zsre"
    exit 1
fi

# ------------------- SSH 连接信息 -------------------
SSH_HOST="wentao@10.16.15.67"
SSH_PORT="22"

# ------------------- 确定远程路径 -------------------
# 本地仓库根目录（Windows 风格转 Git Bash）
LOCAL_REPO_ROOT="d:/大山中学/2026春/大语言模型知识编辑/knowledge_sequence_editing"

# ------------------- 模型映射 -------------------
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
        echo "[错误] 不支持的模型: $MODEL"
        exit 1
        ;;
esac

# ------------------- 算法专属配置 -------------------
if [[ "$ALG" == "MEMIT" ]]; then
    CUDA_DEV=2
    CONDA_ENV="memit_dqg"
    REMOTE_SCRIPT="/tmp/run_${ALG}_${MODEL}_${DS}_${NUM}.sh"
    RUN_DIR="~/xzw1/dqg/memit"
    PYTHON_CMD="python -m experiments.evaluate \
        --alg_name MEMIT \
        --model_name $MODEL_PATH \
        --hparams_fname $HPARAMS_FNAME \
        --num_edits $NUM \
        --ds_name $DS \
        --skip_generation_tests"
else
    CUDA_DEV=3
    CONDA_ENV="pmet_dqg"
    REMOTE_SCRIPT="/tmp/run_${ALG}_${MODEL}_${DS}_${NUM}.sh"
    RUN_DIR="~/xzw1/dqg/PMET/edit"
    PYTHON_CMD="python evaluate.py \
        --alg_name PMET \
        --model_name $MODEL_NAME_CHOICE \
        --model_path $MODEL_PATH \
        --hparams_fname $HPARAMS_FNAME \
        --num_edits $NUM \
        --ds_name $DS \
        --skip_generation_tests"
fi

# ------------------- 打印配置 -------------------
echo "=========================================="
echo "  算法:    $ALG"
echo "  模型:    $MODEL"
echo "  HPARAMS: $HPARAMS_FNAME"
echo "  数据集:  $DS"
echo "  编辑数:  $NUM"
echo "  CUDA:    $CUDA_DEV"
echo "  Conda:   $CONDA_ENV"
echo "  目标目录: $RUN_DIR"
echo "=========================================="

# ------------------- 步骤 1: Git 推送 -------------------
echo ""
echo "[步骤 1/3] 本地 Git 推送..."
cd "$LOCAL_REPO_ROOT"
git add -A
git commit -m "auto: 准备运行 $ALG $MODEL $DS $NUM" 2>/dev/null || true
git push origin main 2>/dev/null || git push origin master 2>/dev/null || true
echo "[完成] Git 已推送"

# ------------------- 步骤 2: 服务器拉取 + rsync -------------------
echo ""
echo "[步骤 2/3] 服务器拉取最新代码 + rsync..."
ssh -p $SSH_PORT $SSH_HOST << 'SSH_EOF'
set -e
cd ~/xzw1/dqg/memit && git pull origin main 2>/dev/null || git pull origin master 2>/dev/null || true
cd ~/xzw1/dqg/PMET && git pull origin main 2>/dev/null || git pull origin master 2>/dev/null || true
# rsync 同步实验目录（排除 .git 和大文件）
echo "同步 MEMIT..."
rsync -av --exclude='.git' --exclude='__pycache__' ~/xzw1/dqg/memit/ /home/wentao/xzw1/dqg/memit/ 2>/dev/null || true
echo "同步 PMET..."
rsync -av --exclude='.git' --exclude='__pycache__' ~/xzw1/dqg/PMET/ /home/wentao/xzw1/dqg/PMET/ 2>/dev/null || true
echo "[完成] 代码已同步"
SSH_EOF

# ------------------- 步骤 3: 构建远程脚本并执行 -------------------
echo ""
echo "[步骤 3/3] 生成远程执行脚本..."

# 在服务器上创建并执行脚本
ssh -p $SSH_PORT $SSH_HOST << EOF
set -e
cat > $REMOTE_SCRIPT << 'INNER_EOF'
#!/bin/bash
cd $RUN_DIR
unset SSL_CERT_FILE
unset CURL_CA_BUNDLE
export CUDA_VISIBLE_DEVICES=$CUDA_DEV
source /root/miniconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV
echo "========== 开始运行 =========="
echo "算法: $ALG"
echo "模型: $MODEL ($MODEL_PATH)"
echo "数据集: $DS"
echo "编辑数: $NUM"
echo "CUDA: $CUDA_DEV"
echo "==========================="
$PYTHON_CMD
INNER_EOF
chmod +x $REMOTE_SCRIPT
echo "执行脚本已生成: $REMOTE_SCRIPT"
echo "开始执行..."
bash $REMOTE_SCRIPT
EOF

echo ""
echo "=========================================="
echo " 实验已提交到服务器"
echo " MEMIT → CUDA $CUDA_DEV"
echo " PMET  → CUDA $CUDA_DEV"
echo " 结果保存在服务器 ~/xzw1/dqg/ 下的对应目录"
echo "=========================================="
