#!/bin/bash
#SBATCH --job-name=cdpg         # Nombre del trabajo
#SBATCH --output=outputs/cdpg_%j.out         # Nombre del archivo de salida
#SBATCH --error=outputs/cdpg_%j.err          # Nombre del archivo de error
#SBATCH --cpus-per-task=1             # Número de CPUs por tarea
#SBATCH --mem=10G                      # Memoria por nodo
#SBATCH --partition=dgx           # Cola (partición) a la que enviar el trabajo
##SBATCH --nodelist=freezer.iuii.ua.es
#SBATCH --gres=gpu:1
##SBATCH --gres=shard:9
#SBATCH --time=12:00:00
##SBATCH --array=3-14%4
##SBATCH --dependency=afterany:14119

# Aquí empieza la sección de comandos que se van a ejecutar

echo "Iniciando trabajo en `hostname` a las `date`, con device = [$SLURM_IDX] y SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"

SRC_DIR=/home/grfia/ffuentes/Scratch/ssmt/

cd $SRC_DIR
source ~/.uvenvs/searchsmt/bin/activate

echo "python create_dataset.py $1"
python create_dataset.py $1