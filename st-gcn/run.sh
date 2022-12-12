#!/bin/bash
#SBATCH -J sk-train
#SBATCH -p CS272
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=all
#SBATCH --mail-user=YOU@MAIL.COM
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=../train_sk.out
#SBATCH --error=../train_sk.err

config_path=./config/st_gcn/FineDiving-skeleton/train.yaml
work_dir=./work_dir
python main.py recognition -c ${config_path} --work_dir ${work_dir}