#!/bin/sh
#SBATCH -p a100
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -c 16
#SBATCH --time=24:00:00

module load conda/python3
conda activate tunnel


if [ "$SLURM_ARRAY_TASK_ID" -eq "1" ]; then
	python run_analysis.py --root /scratch/jsh2/datasets --output /scratch/jsh2/tunnel/results /scratch/jsh2/tunnel/models/resnet18*seed0*
elif [ "$SLURM_ARRAY_TASK_ID" -eq "2" ]; then
	python run_analysis.py --root /scratch/jsh2/datasets --output /scratch/jsh2/tunnel/results /scratch/jsh2/tunnel/models/resnet18*seed1*
elif [ "$SLURM_ARRAY_TASK_ID" -eq "3" ]; then
	python run_analysis.py --root /scratch/jsh2/datasets --output /scratch/jsh2/tunnel/results /scratch/jsh2/tunnel/models/resnet18*seed2*
elif [ "$SLURM_ARRAY_TASK_ID" -eq "4" ]; then
	python run_analysis.py --root /scratch/jsh2/datasets --output /scratch/jsh2/tunnel/results /scratch/jsh2/tunnel/models/resnet34*seed0*
elif [ "$SLURM_ARRAY_TASK_ID" -eq "5" ]; then
	python run_analysis.py --root /scratch/jsh2/datasets --output /scratch/jsh2/tunnel/results /scratch/jsh2/tunnel/models/resnet34*seed1*
elif [ "$SLURM_ARRAY_TASK_ID" -eq "6" ]; then
	python run_analysis.py --root /scratch/jsh2/datasets --output /scratch/jsh2/tunnel/results /scratch/jsh2/tunnel/models/resnet34*seed2*
elif [ "$SLURM_ARRAY_TASK_ID" -eq "7" ]; then
	python run_analysis.py --root /scratch/jsh2/datasets --output /scratch/jsh2/tunnel/results /scratch/jsh2/tunnel/models/vgg19*seed0*
elif [ "$SLURM_ARRAY_TASK_ID" -eq "8" ]; then
	python run_analysis.py --root /scratch/jsh2/datasets --output /scratch/jsh2/tunnel/results /scratch/jsh2/tunnel/models/vgg19*seed1*
elif [ "$SLURM_ARRAY_TASK_ID" -eq "9" ]; then
	python run_analysis.py --root /scratch/jsh2/datasets --output /scratch/jsh2/tunnel/results /scratch/jsh2/tunnel/models/vgg19*seed2*
fi

