#!/bin/bash
#SBATCH --partition=gpunodes 
#SBATCH -c 24
#SBATCH --mem=40G 
#SBATCH --gres=gpu:rtx_a6000:1 
#SBATCH -t 1-0


source /w/284/jameschen/.venv/bin/activate\

cd src

python train.py

deactivate                       