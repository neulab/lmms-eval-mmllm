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

# python3 evaluate.py --input_file "./responses/llava-1.5-7b-hf.json"
# python3 evaluate.py --input_file "./responses/llava-v1.6-vicuna-7b-hf.json"

# python3 evaluate.py --response_file "./responses/korean_prompts_responses.json" --input_file "./responses/llava-1.5-7b-hf-Korean.json"
# python3 evaluate.py --response_file "./responses/chinese_prompts_responses.json" --input_file "./responses/llava-1.5-7b-hf-Chinese.json"
# python3 evaluate.py --response_file "./responses/english_prompts_responses.json" --input_file "./responses/llava-1.5-7b-hf-English.json"
# python3 evaluate.py --response_file "./responses/japanese_prompts_responses.json" --input_file "./responses/llava-1.5-7b-hf-Japanese.json"
# python3 evaluate.py --response_file "./responses/hindi_prompts_responses.json" --input_file "./responses/llava-1.5-7b-hf-Hindi.json"
# python3 evaluate.py --response_file "./responses/spanish_prompts_responses.json" --input_file "./responses/llava-1.5-7b-hf-Spanish.json"
# python3 evaluate.py --response_file "./responses/indonesian_prompts_responses.json" --input_file "./responses/llava-1.5-7b-hf-Indonesian.json"
# python3 evaluate.py --response_file "./responses/kinyarwanda_prompts_responses.json" --input_file "./responses/llava-1.5-7b-hf-Kinyarwanda.json"

# python3 evaluate.py --response_file "./responses/llava-1.5-7b-hf-Korean.json" --input_file "./responses/llava-1.5-7b-hf-Korean.json"
# python3 evaluate.py --response_file "./responses/llava-1.5-7b-hf-Chinese.json" --input_file "./responses/llava-1.5-7b-hf-Chinese.json"
# python3 evaluate.py --response_file "./responses/llava-1.5-7b-hf-English.json" --input_file "./responses/llava-1.5-7b-hf-English.json"
# python3 evaluate.py --response_file "./responses/llava-1.5-7b-hf-Japanese.json" --input_file "./responses/llava-1.5-7b-hf-Japanese.json"
# python3 evaluate.py --response_file "./responses/llava-1.5-7b-hf-Hindi.json" --input_file "./responses/llava-1.5-7b-hf-Hindi.json"
# python3 evaluate.py --response_file "./responses/llava-1.5-7b-hf-Spanish.json" --input_file "./responses/llava-1.5-7b-hf-Spanish.json"
# python3 evaluate.py --response_file "./responses/llava-1.5-7b-hf-Indonesian.json" --input_file "./responses/llava-1.5-7b-hf-Indonesian.json"
# python3 evaluate.py --response_file "./responses/llava-1.5-7b-hf-Kinyarwanda.json" --input_file "./responses/llava-1.5-7b-hf-Kinyarwanda.json"

python3 evaluate.py --response_file "./responses/llava-v1.6-vicuna-7b-hf-Korean.json" --input_file "./responses/llava-1.5-7b-hf-Korean.json"
python3 evaluate.py --response_file "./responses/llava-v1.6-vicuna-7b-hf-Chinese.json" --input_file "./responses/llava-1.5-7b-hf-Chinese.json"
python3 evaluate.py --response_file "./responses/llava-v1.6-vicuna-7b-hf-English.json" --input_file "./responses/llava-1.5-7b-hf-English.json"
python3 evaluate.py --response_file "./responses/llava-v1.6-vicuna-7b-hf-Japanese.json" --input_file "./responses/llava-1.5-7b-hf-Japanese.json"
python3 evaluate.py --response_file "./responses/llava-v1.6-vicuna-7b-hf-Hindi.json" --input_file "./responses/llava-1.5-7b-hf-Hindi.json"
python3 evaluate.py --response_file "./responses/llava-v1.6-vicuna-7b-hf-Spanish.json" --input_file "./responses/llava-1.5-7b-hf-Spanish.json"
python3 evaluate.py --response_file "./responses/llava-v1.6-vicuna-7b-hf-Indonesian.json" --input_file "./responses/llava-1.5-7b-hf-Indonesian.json"
python3 evaluate.py --response_file "./responses/llava-v1.6-vicuna-7b-hf-Kinyarwanda.json" --input_file "./responses/llava-1.5-7b-hf-Kinyarwanda.json"


# python3 evaluate.py --input_file "./responses/llava-1.5-7b-hf-Korean.json"
# python3 evaluate.py --input_file "./responses/llava-v1.6-vicuna-7b-hf-Korean.json"

# python3 evaluate.py --input_file "./responses/llava-1.5-7b-hf-Chinese.json"
# python3 evaluate.py --input_file "./responses/llava-v1.6-vicuna-7b-hf-Chinese.json"

# python3 evaluate.py --input_file "./responses/llava-v1.6-vicuna-7b-hf-Korean.json" --response_file "./responses/llava-1.5-vicuna-7b-v0.3-Korean.json"
# python3 evaluate.py --input_file "./responses/llava-v1.6-vicuna-7b-hf-Chinese.json" --response_file "./responses/llava-1.5-vicuna-7b-v0.3-Chinese.json"
# python3 evaluate.py --input_file "./responses/llava-v1.6-vicuna-7b-hf-English.json" --response_file "./responses/llava-1.5-vicuna-7b-v0.3-English.json"