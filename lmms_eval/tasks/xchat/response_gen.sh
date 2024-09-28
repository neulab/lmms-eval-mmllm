#!/bin/bash
#SBATCH --job-name=seungone-xchat
#SBATCH --output=xchat.out
#SBATCH --error=xchat.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:A6000:1
#SBATCH --mail-user=seungone@cmu.edu
#SBATCH --mail-type=START,END,FAIL

echo $SLURM_JOB_ID

__conda_setup="$('/home/seungonk/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
eval "$__conda_setup"
conda activate base

export TRANSFORMERS_CACHE="/data/tir/projects/tir7/user_data/seungonk/huggingface"
export HF_CACHE="/data/tir/projects/tir7/user_data/seungonk/huggingface"

# English
# python3 response_gen.py --model_name "llava-hf/llava-1.5-7b-hf" --language "English" --output_file "./responses/llava-1.5-7b-hf-English.json" --gpu_num 1
# python3 response_gen.py --model_name "llava-hf/llava-v1.6-vicuna-7b-hf" --language "English" --output_file "./responses/llava-v1.6-vicuna-7b-hf-English.json" --gpu_num 1

# Chinese
# python3 response_gen.py --model_name "llava-hf/llava-1.5-7b-hf" --language "Chinese" --output_file "./responses/llava-1.5-7b-hf-Chinese.json" --gpu_num 1
# python3 response_gen.py --model_name "llava-hf/llava-v1.6-vicuna-7b-hf" --language "Chinese" --output_file "./responses/llava-v1.6-vicuna-7b-hf-Chinese.json" --gpu_num 1

# Hindi
# python3 response_gen.py --model_name "llava-hf/llava-1.5-7b-hf" --language "Hindi" --output_file "./responses/llava-1.5-7b-hf-Hindi.json" --gpu_num 1
# python3 response_gen.py --model_name "llava-hf/llava-v1.6-vicuna-7b-hf" --language "Hindi" --output_file "./responses/llava-v1.6-vicuna-7b-hf-Hindi.json" --gpu_num 1

# Indonesian
# python3 response_gen.py --model_name "llava-hf/llava-1.5-7b-hf" --language "Indonesian" --output_file "./responses/llava-1.5-7b-hf-Indonesian.json" --gpu_num 1
python3 response_gen.py --model_name "llava-hf/llava-v1.6-vicuna-7b-hf" --language "Indonesian" --output_file "./responses/llava-v1.6-vicuna-7b-hf-Indonesian.json" --gpu_num 1

# Japanese
# python3 response_gen.py --model_name "llava-hf/llava-1.5-7b-hf" --language "Japanese" --output_file "./responses/llava-1.5-7b-hf-Japanese.json" --gpu_num 1
python3 response_gen.py --model_name "llava-hf/llava-v1.6-vicuna-7b-hf" --language "Japanese" --output_file "./responses/llava-v1.6-vicuna-7b-hf-Japanese.json" --gpu_num 1

# Kinyarwanda
# python3 response_gen.py --model_name "llava-hf/llava-1.5-7b-hf" --language "Kinyarwanda" --output_file "./responses/llava-1.5-7b-hf-Kinyarwanda.json" --gpu_num 1
python3 response_gen.py --model_name "llava-hf/llava-v1.6-vicuna-7b-hf" --language "Kinyarwanda" --output_file "./responses/llava-v1.6-vicuna-7b-hf-Kinyarwanda.json" --gpu_num 1

# Korean
# python3 response_gen.py --model_name "llava-hf/llava-1.5-7b-hf" --language "Korean" --output_file "./responses/llava-1.5-7b-hf-Korean.json" --gpu_num 1
python3 response_gen.py --model_name "llava-hf/llava-v1.6-vicuna-7b-hf" --language "Korean" --output_file "./responses/llava-v1.6-vicuna-7b-hf-Korean.json" --gpu_num 1

# Spanish
# python3 response_gen.py --model_name "llava-hf/llava-1.5-7b-hf" --language "Spanish" --output_file "./responses/llava-1.5-7b-hf-Spanish.json" --gpu_num 1
python3 response_gen.py --model_name "llava-hf/llava-v1.6-vicuna-7b-hf" --language "Spanish" --output_file "./responses/llava-v1.6-vicuna-7b-hf-Spanish.json" --gpu_num 1