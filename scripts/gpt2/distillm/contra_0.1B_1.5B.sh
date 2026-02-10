#! /bin/bash

# =========================================================
# 1. CẤU HÌNH MẠNG & GPU
# =========================================================
MASTER_ADDR=localhost
MASTER_PORT=2012
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=2  # Set mặc định là 2 (cho 2 GPU A100 của bạn)

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

# =========================================================
# 2. CẤU HÌNH ĐƯỜNG DẪN & MODEL
# =========================================================
# Lấy thư mục hiện tại làm gốc
BASE_PATH=${1:-$(pwd)}

# --- STUDENT (GPT-2 120M) ---
CKPT_NAME="gpt2-120M-dolly"
CKPT="bachthetrollface/gpt2-120M-init-dolly"

# --- TEACHER (GPT-2 1.5B) ---
TEACHER_CKPT_NAME="gpt2-1.5B-dolly"
TEACHER_CKPT="bachthetrollface/gpt2-1.5B-teacher-dolly"

# --- VELOCITY FIELD & PROJECTOR (Từ bước train trước) ---
# Lưu ý: Đảm bảo file .pth nằm đúng trong thư mục này
VF_SAVE_DIR="${BASE_PATH}/results/gpt2/train/velocity_field"
VELOCITY_FIELD_CKPT="${VF_SAVE_DIR}/velocity_field.pth"
PROJECTOR_CKPT="${VF_SAVE_DIR}/projector.pth"

# --- DATA ---
# Dữ liệu DistiLLM-2 đã format ở bước trước
DATA_DIR="${BASE_PATH}/processed_data/dolly/full/gpt2/"

# Dữ liệu OpenWebText cho LM loss (Tùy chọn - Comment nếu không có)
LM_DATA_DIR="${BASE_PATH}/processed_data/openwebtext/gpt2/512/22.87K/"

# =========================================================
# 3. HYPERPARAMETERS
# =========================================================
BATCH_SIZE=16       # An toàn cho 2 GPU
LR=0.0005
GRAD_ACC=2         # Tăng tích lũy gradient để bù cho batch size nhỏ
EVAL_BATCH_SIZE=32
MAX_LENGTH=512
SAVE_PATH="${BASE_PATH}/results/gpt2/train/contra/distillm/0.1B_1.5B"
SEED=10

# =========================================================
# 4. TẠO OPTIONS
# =========================================================
OPTS=""
# Model Config
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --teacher-model-path ${TEACHER_CKPT}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --teacher-ckpt-name ${TEACHER_CKPT_NAME}"
OPTS+=" --teacher-model-fp16"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"

# [QUAN TRỌNG] Tắt Gradient Checkpointing để tránh lỗi với DeepSpeed + Frozen Modules
OPTS+=" --gradient-checkpointing" 

# Data Config
OPTS+=" --data-dir ${DATA_DIR}"
if [ -n "$LM_DATA_DIR" ]; then
    OPTS+=" --lm-data-dir ${LM_DATA_DIR}"
fi
OPTS+=" --num-workers 4"
OPTS+=" --dev-num 1000"

# HP Config
OPTS+=" --lr ${LR}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --warmup-iters 0"
OPTS+=" --lr-decay-style cosine"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 1.0"
OPTS+=" --epochs 20"
OPTS+=" --kd-ratio 1.0"

# Length Config
OPTS+=" --max-length ${MAX_LENGTH}"
OPTS+=" --max-prompt-length 256"

# Runtime Config
OPTS+=" --do-train"
OPTS+=" --do-valid"
OPTS+=" --eval-gen"
OPTS+=" --save-interval -1"
OPTS+=" --eval-interval -1"
OPTS+=" --log-interval 10"
OPTS+=" --mid-log-num -1"
OPTS+=" --save ${SAVE_PATH}"
OPTS+=" --seed ${SEED}"

# DeepSpeed
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config.json"

# Type: Adaptive Skewed Reverse KL + Contra
OPTS+=" --type adaptive-srkl-contra"
OPTS+=" --skew-alpha 0.1"

# Generation Config
OPTS+=" --do-sample"
OPTS+=" --top-k 0"
OPTS+=" --top-p 1.0"
OPTS+=" --temperature 1.0"

# DistiLLM Config
OPTS+=" --student-gen"
OPTS+=" --gen-num-beams 1"
OPTS+=" --gen-top-p 1.0"
OPTS+=" --init-threshold 0.0"
OPTS+=" --loss-eps 0.1"
OPTS+=" --capacity 1000"

# Contra-KD Config
OPTS+=" --velocity-n-layers 4"
OPTS+=" --velocity-d-model 1024"
OPTS+=" --d-teacher 1600"
OPTS+=" --d-student 768"
OPTS+=" --num-distill-layers 6"
OPTS+=" --num-teacher-layers 48"
OPTS+=" --num-student-layers 12"
OPTS+=" --teacher-device 0"
OPTS+=" --student-device 0"
OPTS+=" --velocity-field-path ${VELOCITY_FIELD_CKPT}"
OPTS+=" --projector-path ${PROJECTOR_CKPT}"
OPTS+=" --velocity-epochs 1"
OPTS+=" --velocity-update-interval 1"

# =========================================================
# 5. MÔI TRƯỜNG & CHẠY LỆNH
# =========================================================
export NCCL_DEBUG=""
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}

# Fix lỗi thư viện GLIBCXX (từ bước trước)
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/hungpv/miniconda3/envs/warp/lib

# Fix lỗi OOM
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/finetune.py ${OPTS} $@"

echo "-------------------------------------------------------"
echo "Starting Contra-KD Finetuning"
echo "Student: ${CKPT}"
echo "Velocity Field: ${VELOCITY_FIELD_CKPT}"
echo "Data: ${DATA_DIR}"
echo "Save Path: ${SAVE_PATH}"
echo "-------------------------------------------------------"
echo ${CMD}

mkdir -p ${SAVE_PATH}
CODE_BASE=HF ${CMD}