#!/bin/bash
#SBATCH -J test
#SBATCH -p CS272
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=all
#SBATCH --mail-user=YOU@MAIL.COM
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=../test.out
#SBATCH --error=../test.err
#SBATCH --exclude=ai_gpu[08-25]


#../result_txt/mult_st_gcn/train_sk_1.out
#../result_txt/mult_st_gcn/train_sk_1.err

# ../result_txt/st_gcn/train_sk_1.out
# ../result_txt/st_gcn/train_sk_1.err



config_path=./config/st_gcn/FineDiving-skeleton/test.yaml
work_dir=./work_dir/st_gcn

# config_path=./config/st_gcn/FineDiving-skeleton/train_mult.yaml
# work_dir=./work_dir/st_gcn_mult

python main.py recognition -c ${config_path} --work_dir ${work_dir}