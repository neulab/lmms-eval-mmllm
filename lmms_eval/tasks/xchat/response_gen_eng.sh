#!/bin/bash
#SBATCH --job-name=seungone-response_gen_eng
#SBATCH --output=response_gen_eng.out
#SBATCH --error=response_gen_eng.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:A6000:4
#SBATCH --mail-user=seungone@cmu.edu
#SBATCH --mail-type=START,END,FAIL

echo $SLURM_JOB_ID

__conda_setup="$('/home/seungonk/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
eval "$__conda_setup"
conda activate base

export TRANSFORMERS_CACHE="/data/tir/projects/tir7/user_data/seungonk/huggingface"
export HF_CACHE="/data/tir/projects/tir7/user_data/seungonk/huggingface"

python3 response_gen.py --model_name "seungone/final-general-gpt4o-mini-10000-llama3_8b" --base_model_name "llama3" --domain "general" --output_file "/home/seungonk/benchmarking-data-gen/responses/final-general-gpt4o-mini-10000-llama3_8b.json" --gpu_num 4
python3 response_gen.py --model_name "seungone/final-math-gpt4o-mini-10000-llama3_8b" --base_model_name "llama3" --domain "math" --output_file "/home/seungonk/benchmarking-data-gen/responses/final-math-gpt4o-mini-10000-llama3_8b.json" --gpu_num 4
python3 response_gen.py --model_name "seungone/final-code-gpt4o-mini-10000-llama3_8b" --base_model_name "llama3" --domain "code" --output_file "/home/seungonk/benchmarking-data-gen/responses/final-code-gpt4o-mini-10000-llama3_8b.json" --gpu_num 4