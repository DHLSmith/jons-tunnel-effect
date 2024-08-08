#!/bin/bash -l
#SBATCH -p a100
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -c 16
#SBATCH --time=24:00:00
#SBATCH --get-user-env=L

module load conda/python3
source activate tunnel

if [ "$SLURM_ARRAY_TASK_ID" -eq "1" ]; then
	python train_model.py --model=resnet18 --dataset=cifar10 --seed=0
elif [ "$SLURM_ARRAY_TASK_ID" -eq "2" ]; then
	python train_model.py --model=resnet18 --dataset=cifar10 --seed=1
elif [ "$SLURM_ARRAY_TASK_ID" -eq "3" ]; then
	python train_model.py --model=resnet18 --dataset=cifar10 --seed=2
elif [ "$SLURM_ARRAY_TASK_ID" -eq "4" ]; then
	python train_model.py --model=resnet34 --dataset=cifar10 --seed=0
elif [ "$SLURM_ARRAY_TASK_ID" -eq "5" ]; then
	python train_model.py --model=resnet34 --dataset=cifar10 --seed=1
elif [ "$SLURM_ARRAY_TASK_ID" -eq "6" ]; then
	python train_model.py --model=resnet34 --dataset=cifar10 --seed=2
elif [ "$SLURM_ARRAY_TASK_ID" -eq "7" ]; then
	python train_model.py --model=vgg19 --dataset=cifar10 --seed=0
elif [ "$SLURM_ARRAY_TASK_ID" -eq "8" ]; then
	python train_model.py --model=vgg19 --dataset=cifar10 --seed=1
elif [ "$SLURM_ARRAY_TASK_ID" -eq "9" ]; then
	python train_model.py --model=vgg19 --dataset=cifar10 --seed=2
fi

