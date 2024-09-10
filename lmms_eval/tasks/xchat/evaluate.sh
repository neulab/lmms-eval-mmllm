#!/bin/bash
#SBATCH --job-name=seungone-evaluate
#SBATCH --output=evaluate.out
#SBATCH --error=evaluate.err
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2:00:00
#SBATCH --mail-user=seungone@cmu.edu
#SBATCH --mail-type=START,END,FAIL

echo $SLURM_JOB_ID

__conda_setup="$('/home/seungonk/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
eval "$__conda_setup"
conda activate base

# export TRANSFORMERS_CACHE="/data/tir/projects/tir7/user_data/seungonk/huggingface"
# export HF_CACHE="/data/tir/projects/tir7/user_data/seungonk/huggingface"

python3 evaluate.py --input_file "./responses/llava-1.5-7b-hf.json"
python3 evaluate.py --input_file "./responses/llava-v1.6-vicuna-7b-hf.json"
# python3 evaluate.py --input_file "./responses/llava-1.5-vicuna-7b-v0.3-hf.json"