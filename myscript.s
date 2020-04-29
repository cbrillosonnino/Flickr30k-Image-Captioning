#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --time=48:00:00
#SBATCH --job-name=image_captioning
#SBATCH --mail-type=END
#SBATCH --mail-user=cbs488@nyu.edu

module purge
module load cudnn/10.1v7.6.5.32 
module load cuda/10.2.89
module load python3/intel/3.7.3

cd /scratch/cbs488/Flickr30k-Image-Captioning

source py3.7.3/bin/activate

python train.py

