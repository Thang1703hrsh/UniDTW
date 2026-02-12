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

# --- PROJECTOR (từ bước train trước) ---
VF_SAVE_DIR="${BASE_PATH}/results/gpt2/train/velocity_field"
PROJECTOR_CKPT="${VF_SAVE_DIR}/projector.pth"

# --- DATA ---
DATA_DIR="${BASE_PATH}/processed_data/dolly/full/gpt2/"

LM_DATA_DIR="${BASE_PATH}/processed_data/openwebtext/gpt2/512/22.87K/"

# =========================================================
# 3. HYPERPARAMETERS
# =========================================================
BATCH_SIZE=8       # An toàn cho 2 GPU
LR=0.0005
GRAD_ACC=2         # Tăng tích lũy gradient để bù cho batch size nhỏ
EVAL_BATCH_SIZE=64
MAX_LENGTH=512
SAVE_PATH="${BASE_PATH}/results/gpt2/train/dtw/fdd/base"
SEED=10

OPTS=""
# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --teacher-model-path ${TEACHER_CKPT}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --teacher-ckpt-name ${TEACHER_CKPT_NAME}"
OPTS+=" --teacher-model-fp16"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --gradient-checkpointing"
# data
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --lm-data-dir ${LM_DATA_DIR}"
OPTS+=" --num-workers 4"
OPTS+=" --dev-num 1000"

# DTW Config
OPTS+=" --dtw-weight 1.0"
OPTS+=" --dtw-window 32"
OPTS+=" --dtw-gamma 0.1"
OPTS+=" --dtw-distance cosine"
OPTS+=" --dtw-normalize"

OPTS+=" --dtw-unitization"
# OPTS+=" --dtw-importance-weights teacher_entropy"

OPTS+=" --num-distill-layers 6"
OPTS+=" --num-teacher-layers 48"
OPTS+=" --num-student-layers 12"

# hp
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
# length
OPTS+=" --max-length ${MAX_LENGTH}"
OPTS+=" --max-prompt-length 256"
# runtime
OPTS+=" --do-train"
OPTS+=" --do-valid"
OPTS+=" --eval-gen"
OPTS+=" --save-interval -1"
OPTS+=" --eval-interval -1"
OPTS+=" --log-interval 10"
OPTS+=" --mid-log-num -1"
OPTS+=" --save ${SAVE_PATH}"
# seed
OPTS+=" --seed ${SEED}"
# deepspeed
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config.json"
# type
OPTS+=" --type rkl-fdd-dtw"
# gen
OPTS+=" --do-sample"
OPTS+=" --top-k 0"
OPTS+=" --top-p 1.0"
OPTS+=" --temperature 1.0"


OPTS+=" --d-teacher 1600"
OPTS+=" --d-student 768"

OPTS+=" --projector-path ${PROJECTOR_CKPT}"

export NCCL_DEBUG=""
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/finetune.py ${OPTS} $@"

echo "-------------------------------------------------------"
echo "Starting DTW FDD Finetuning"
echo "Student: ${CKPT}"
echo "Teacher: ${TEACHER_CKPT}"
echo "Projector: ${PROJECTOR_CKPT}"
echo "Data: ${DATA_DIR}"
echo "Save Path: ${SAVE_PATH}"
echo "-------------------------------------------------------"
echo ${CMD}

mkdir -p ${SAVE_PATH}
CODE_BASE=HF ${CMD}
