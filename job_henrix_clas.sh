#!/bin/bash
#SBATCH -p gpu --gres=gpu:a100:1
#SBATCH --job-name=dna-32k
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4 --mem=250000M
#SBATCH --time=24:00:00

./run_hendrix_classification.sh