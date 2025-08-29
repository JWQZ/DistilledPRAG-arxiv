#!/bin/bash

# ====================== 参数配置 ======================
DATASET_PATH="./data/2wikimultihopqa/train_passages_deduplication_80000_90000.json"
MODEL_PATH="./models/Llama-3.2-1B-Instruct-Doc_mask"
DEVICE="cuda:7"

# ====================== 日志文件命名 ======================
# TIMESTAMP=$(date +"%y_%m_%d_%H_%M")
DATASET_NAME=$(basename "$DATASET_PATH" .json)
LOG_FILE="./logs/logfile_${DATASET_NAME}.txt"
# 提前创建日志文件，防止后续 echo >> 失败
touch "$LOG_FILE"

# ====================== 构建训练命令 ======================
CMD="python runs.py \
  --dataset_path ${DATASET_PATH} \
  --model_path ${MODEL_PATH} \
  --device ${DEVICE}"

# ====================== 启动训练任务 ======================
echo "Launching training..."
echo "$CMD"
eval "nohup $CMD > \"$LOG_FILE\" 2>&1 &"

PID=$!
echo "Training started. PID: $PID"
echo "Training started. Logging to: $LOG_FILE"