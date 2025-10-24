#!/bin/bash
#SBATCH --job-name=ssmt-pg         # Nombre del trabajo
#SBATCH --output=outputs/train_%j.out         # Nombre del archivo de salida
#SBATCH --error=outputs/train_%j.err          # Nombre del archivo de error
#SBATCH --cpus-per-task=1             # Número de CPUs por tarea
#SBATCH --mem=10G                      # Memoria por nodo
#SBATCH --partition=dgx           # Cola (partición) a la que enviar el trabajo
##SBATCH --nodelist=freezer.iuii.ua.es
#SBATCH --gres=gpu:1
##SBATCH --gres=shard:9
#SBATCH --time=48:00:00
##SBATCH --dependency=afterany:14119
##SBATCH --array=3-14,19-21,25-27
##SBATCH --array=53-57
#SBATCH --array=53-62%5

# run again
# 3-8       6
# 9-14      6
# 19-21     3
# 25-27     3

# Aquí empieza la sección de comandos que se van a ejecutar

echo "Iniciando trabajo en `hostname` a las `date`, con device = [$SLURM_IDX]"
nvidia-smi

SRC_DIR=/home/grfia/ffuentes/Scratch/ssmt/

cd $SRC_DIR
source /home/grfia/ffuentes/.uvenvs/searchsmt/bin/activate

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
