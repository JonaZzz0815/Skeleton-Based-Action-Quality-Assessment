#!/bin/bash
#SBATCH -J sk-st_gcn
#SBATCH -p CS272
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=all
#SBATCH --mail-user=YOU@MAIL.COM
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=../train_sk_2.out
#SBATCH --error=../train_sk_2.err

config_path=./config/st_gcn/FineDiving-skeleton/train.yaml
# config_path=./config/st_gcn_with_rgb/FineDiving-skeleton/train.yaml
work_dir=./work_dir/st_gcn
# work_dir=./work_dir/st_gcn_with_rgb
python main.py recognition -c ${config_path} --work_dir ${work_dir}