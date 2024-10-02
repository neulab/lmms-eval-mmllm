#!/bin/bash
#SBATCH --job-name=seungone-convert_to_llava_format_eng
#SBATCH --output=convert_to_llava_format_eng.out
#SBATCH --error=convert_to_llava_format_eng.err
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --mail-user=seungone@cmu.edu
#SBATCH --mail-type=START,END,FAIL

echo $SLURM_JOB_ID

__conda_setup="$('/home/seungonk/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
eval "$__conda_setup"
conda activate base

export TRANSFORMERS_CACHE="/data/tir/projects/tir7/user_data/seungonk/huggingface"
export HF_CACHE="/data/tir/projects/tir7/user_data/seungonk/huggingface"


python3 convert_to_llava_format.py --language "Chinese" --output_file "./prompts/chinese_prompts.jsonl"
python3 convert_to_llava_format.py --language "English" --output_file "./prompts/english_prompts.jsonl"
python3 convert_to_llava_format.py --language "Hindi" --output_file "./prompts/hindi_prompts.jsonl"
python3 convert_to_llava_format.py --language "Indonesian" --output_file "./prompts/indonesian_prompts.jsonl"
python3 convert_to_llava_format.py --language "Japanese" --output_file "./prompts/japanese_prompts.jsonl"
python3 convert_to_llava_format.py --language "Kinyarwanda" --output_file "./prompts/kinyarwanda_prompts.jsonl"
python3 convert_to_llava_format.py --language "Korean" --output_file "./prompts/korean_prompts.jsonl"
python3 convert_to_llava_format.py --language "Spanish" --output_file "./prompts/spanish_prompts.jsonl"