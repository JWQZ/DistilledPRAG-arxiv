#!/bin/bash

# ====================== 参数配置 ======================
EMBEDDING_MODEL_PATH="./models/long-t5-tglobal-base"
LLM_MODEL_PATH="./models/Llama-3.2-1B-Instruct-Doc_mask"
# LLM_MODEL_PATH="./models/Llama-3.2-1B-Instruct"
SAVE_PATH="./models/Llama-3.2-1B-Instruct-Doc_mask-longt5_capt_48"
# SAVE_PATH="./models/Llama-3.2-1B-Instruct-longt5_capt_14"
# DATASET_PATH="./data_aug_deepseek-v3/train"
# DATASET_PATH="./data_aug_Llama-3.2-1B-Instruct-Doc_mask/2wikimultihopqa/train_passages_deduplication_0_90000"  # 如果为空则用默认构造数据
DATASET_PATH="./data_aug_deepseek-v3/2wikimultihopqa/train_passages_deduplication_0_30000"
DATASET2_PATH="./data_aug_deepseek-v3/2wikimultihopqa/train_2passages_deduplication_0_30000"
# DATASET_PATH="./data_aug_deepseek-v3/2wikimultihopqa/train_passages_deduplication_0_30000_v2"
# DATASET2_PATH="./data_aug_deepseek-v3/2wikimultihopqa/train_2passages_deduplication_0_30000_v2"
# DATASET_PATH="None"
# DATASET2_PATH="None"
TRANSLATOR_TYPE="cross-attention-parameter-translator-s" # 可选值: "parameter-translator", "cross-attention-parameter-translator"
LR=1e-4
HIDDEN_LOSS_TYPE="cosine"  # 可选值: "mse", "cosine"
LOGITS_LOSS_TYPE="kl"  # 可选值: "mse", "kl", "cosine"
# ALPHA_ZERO=0.0 #隐藏层最后一层损失系数
# ALPHA=0.5 #隐藏层加权损失系数
# BETA=0.1 #logits对齐损失系数
# GAMA=1.0 #logits CE损失系数
# 0.23 0.16 4.52 4.46
ALPHA_ZERO=0 #隐藏层最后一层损失系数
ALPHA=0 #隐藏层加权损失系数
BETA=0.1 #logits对齐损失系数
GAMA=1.0 #logits CE损失系数
KL_TEMPERATURE=2.0
BATCH_SIZE=4
EPOCHS=1
SCHEDULER_TYPE="linear"  # 可选值: "cosine", "linear", None
LOG_TOOL="wandb" # "tensorboard", "swanlab"， “wandb”
LOG_DESCRIPTION="小模型，加最后一个LN，双9W数据集，0-0-0.1-1损失,1epoch"  #"masktoken，0-0.5-0.1-1损失,lr1e-4,cross-hyper,train-token"
LOG_STEPS=80
SAVING_STEPS=8000
TRAIN_TOKEN=0
DEVICE_TRANSLATOR="cuda:7"
DEVICE_EMBEDDING="cuda:7"
DEVICE_LLM="cuda:7"

# ====================== 日志文件命名 ======================
TIMESTAMP=$(date +"%y_%m_%d_%H_%M")
LOG_FILE="./logs/logfile_${TIMESTAMP}.txt"
# 提前创建日志文件，防止后续 echo >> 失败
touch "$LOG_FILE"

# ====================== 构建训练命令 ======================
CMD="python train_my_dyprag.py \
  --embedding_model_path ${EMBEDDING_MODEL_PATH} \
  --llm_model_path ${LLM_MODEL_PATH} \
  --save_path ${SAVE_PATH} \
  --dataset_path ${DATASET_PATH} \
  --dataset2_path ${DATASET2_PATH} \
  --translator_type ${TRANSLATOR_TYPE} \
  --lr ${LR} \
  --hidden_loss_type ${HIDDEN_LOSS_TYPE} \
  --logits_loss_type ${LOGITS_LOSS_TYPE} \
  --alpha_zero ${ALPHA_ZERO} \
  --alpha ${ALPHA} \
  --beta ${BETA} \
  --gama ${GAMA} \
  --kl_temperature ${KL_TEMPERATURE} \
  --batch_size ${BATCH_SIZE} \
  --epochs ${EPOCHS} \
  --scheduler_type ${SCHEDULER_TYPE} \
  --log_tool ${LOG_TOOL} \
  --log_description "${LOG_DESCRIPTION}" \
  --log_steps ${LOG_STEPS} \
  --saving_steps ${SAVING_STEPS} \
  --device_translator ${DEVICE_TRANSLATOR} \
  --device_embedding ${DEVICE_EMBEDDING} \
  --device_llm ${DEVICE_LLM} \
  --train_token ${TRAIN_TOKEN}"

# ====================== 启动训练任务 ======================
echo "Launching training..."
echo "$CMD"
eval "nohup $CMD > \"$LOG_FILE\" 2>&1 &"

PID=$!
echo "Training started. PID: $PID"
echo "Training started. Logging to: $LOG_FILE"