#!/bin/bash

# run_and_terminate.sh

set -e

echo "Starting execution..."

# Run your main script
#python your_training_script.py
SRC_DIR= /root/vqa-pg/ #/home/grfia/ffuentes/Scratch/ssmt/

cd $SRC_DIR
source /root/.uvenvs/vqa-pg/bin/activate

#SLURM_ARRAY_TASK_ID=$1
# dataset     learning_rate   batch_size  model_size  resolution  use_lora    lora_rank   trainVT     quant
cfg=$(sed -n "$SLURM_ARRAY_TASK_ID"p ${SRC_DIR}/exps_configs.txt)
dataset=$(echo $cfg | cut -f 1 -d ' ')
learning_rate=$(echo $cfg | cut -f 2 -d ' ')
batch_size=$(echo $cfg | cut -f 3 -d ' ')
model_size=$(echo $cfg | cut -f 4 -d ' ')
resolution=$(echo $cfg | cut -f 5 -d ' ')
use_lora=$(echo $cfg | cut -f 6 -d ' ')
lora_rank=$(echo $cfg | cut -f 7 -d ' ')
trainVT=$(echo $cfg | cut -f 8 -d ' ')
quant=$(echo $cfg | cut -f 9 -d ' ')
epochs=$(echo $cfg | cut -f 10 -d ' ')

echo "train.py \
        --dataset ${dataset} \
        --learning_rate ${learning_rate} \
        --batch_size ${batch_size} \
        --model_size ${model_size} \
        --resolution ${resolution} \
        --use_lora ${use_lora} \
        --lora_rank ${lora_rank} \
        --train_vision_tower ${trainVT} \
        --quant ${quant} \
        --epochs ${epochs}"

echo "Starting training..."

python train.py \
        --dataset ${dataset} \
        --learning_rate ${learning_rate} \
        --batch_size ${batch_size} \
        --model_size ${model_size} \
        --resolution ${resolution} \
        --use_lora ${use_lora} \
        --lora_rank ${lora_rank} \
        --train_vision_tower ${trainVT} \
        --quant ${quant} \
        --epochs ${epochs}

echo "Training finished. Starting testing..."

python test_ood.py \
        --dataset ${dataset} \
        --learning_rate ${learning_rate} \
        --batch_size ${batch_size} \
        --model_size ${model_size} \
        --resolution ${resolution} \
        --use_lora ${use_lora} \
        --lora_rank ${lora_rank} \
        --train_vision_tower ${trainVT} \
        --quant ${quant} \
        --epochs ${epochs}


echo "Testing finished. Terminating pod..."

# Terminate using curl
curl -X POST \
  https://api.runpod.io/graphql \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -d "{\"query\": \"mutation { podTerminate(input: {podId: \\\"${RUNPOD_POD_ID}\\\"}) }\"}"

echo "Pod termination initiated."