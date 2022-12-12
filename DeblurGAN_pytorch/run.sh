#!/bin/bash
#SBATCH -J test
#SBATCH -p CS272
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=all
#SBATCH --mail-user=YOU@MAIL.COM
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=../test_deblur.out
#SBATCH --error=../test_deblur.err

#python deblur_image.py --blurred ../FineDiving_Dataset/Trimmed_Video_Frames/FINADiving_MTL_256s/03/1/ --deblurred ../deblur --resume checkpoint-epoch300.pth

for i in $(seq 1 200)
do 
    echo $i;
    python3 extract_skeleton.py
done

