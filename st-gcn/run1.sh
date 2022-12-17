#!/bin/bash
#SBATCH -J sk-rgb
#SBATCH -p CS272
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=all
#SBATCH --mail-user=YOU@MAIL.COM
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=../train_sk_rgb.out
#SBATCH --error=../train_sk_rgb.err


config_path=./config/st_gcn_with_rgb/FineDiving-skeleton/train.yaml
work_dir=./work_dir/st_gcn_with_rgb

# config_path=./config/st_gcn_with_rgb/FineDiving-skeleton/train_mult.yaml
# work_dir=./work_dir/st_gcn_with_rgb_mult


python main.py recognition -c ${config_path} --work_dir ${work_dir}
# python main.py recognition -c ./config/st_gcn_with_rgb/FineDiving-skeleton/train.yaml --work_dir ./work_dir/st_gcn_with_rgb