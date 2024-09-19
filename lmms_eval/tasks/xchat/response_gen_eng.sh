#!/bin/bash
#SBATCH --job-name=seungone-response_gen_eng
#SBATCH --output=response_gen_eng.out
#SBATCH --error=response_gen_eng.err
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:A6000:2
#SBATCH --mail-user=seungone@cmu.edu
#SBATCH --mail-type=START,END,FAIL

echo $SLURM_JOB_ID

__conda_setup="$('/home/seungonk/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
eval "$__conda_setup"
conda activate base

export TRANSFORMERS_CACHE="/data/tir/projects/tir7/user_data/seungonk/huggingface"
export HF_CACHE="/data/tir/projects/tir7/user_data/seungonk/huggingface"

# python3 response_gen.py --model_name "llava-hf/llava-1.5-7b-hf" --language "English" --output_file "./responses/llava-1.5-7b-hf.json" --gpu_num 2
# python3 response_gen.py --model_name "llava-hf/llava-v1.6-vicuna-7b-hf" --language "English" --output_file "./responses/llava-v1.6-vicuna-7b-hf.json" --gpu_num 2
# python3 response_gen.py --model_name "neulab/llava-1.5-vicuna-7b-v0.3-hf" --language "English" --output_file "./responses/llava-1.5-vicuna-7b-v0.3-hf.json" --gpu_num 2

# python3 response_gen.py --model_name "llava-hf/llava-1.5-7b-hf" --language "English" --output_file "./responses/llava-1.5-7b-hf-English.json" --gpu_num 2
# python3 response_gen.py --model_name "llava-hf/llava-v1.6-vicuna-7b-hf" --language "English" --output_file "./responses/llava-v1.6-vicuna-7b-hf-English.json" --gpu_num 2

python3 response_gen.py --model_name "llava-hf/llava-1.5-7b-hf" --language "Chinese" --output_file "./responses/llava-1.5-7b-hf-Chinese.json" --gpu_num 2
python3 response_gen.py --model_name "llava-hf/llava-v1.6-vicuna-7b-hf" --language "Chinese" --output_file "./responses/llava-v1.6-vicuna-7b-hf-Chinese.json" --gpu_num 2

# python3 response_gen.py --model_name "llava-hf/llava-1.5-7b-hf" --language "Korean" --output_file "./responses/llava-1.5-7b-hf-Korean.json" --gpu_num 2
# python3 response_gen.py --model_name "llava-hf/llava-v1.6-vicuna-7b-hf" --language "Korean" --output_file "./responses/llava-v1.6-vicuna-7b-hf-Korean.json" --gpu_num 2

# python3 evaluate.py --input_file "./responses/llava-1.5-7b-hf.json"
# python3 evaluate.py --input_file "./responses/llava-v1.6-vicuna-7b-hf.json"
# python3 evaluate.py --input_file "./responses/llava-1.5-vicuna-7b-v0.3-hf.json"