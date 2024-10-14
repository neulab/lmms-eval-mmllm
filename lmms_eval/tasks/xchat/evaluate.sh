#!/bin/bash
#SBATCH --job-name=seungone-evaluate
#SBATCH --output=evaluate.out
#SBATCH --error=evaluate.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mail-user=seungone@cmu.edu
#SBATCH --mail-type=START,END,FAIL

echo $SLURM_JOB_ID

__conda_setup="$('/home/seungonk/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
eval "$__conda_setup"
conda activate base

export TRANSFORMERS_CACHE="/data/tir/projects/tir7/user_data/seungonk/huggingface"
export HF_CACHE="/data/tir/projects/tir7/user_data/seungonk/huggingface"

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

# python3 evaluate.py --response_file "./responses/palo-korean.json" --input_file "./responses/llava-1.5-7b-hf-Korean.json"
# python3 evaluate.py --response_file "./responses/palo-chinese.json" --input_file "./responses/llava-1.5-7b-hf-Chinese.json"
# python3 evaluate.py --response_file "./responses/palo-english.json" --input_file "./responses/llava-1.5-7b-hf-English.json"
# python3 evaluate.py --response_file "./responses/palo-japanese.json" --input_file "./responses/llava-1.5-7b-hf-Japanese.json"
# python3 evaluate.py --response_file "./responses/palo-hindi.json" --input_file "./responses/llava-1.5-7b-hf-Hindi.json"
# python3 evaluate.py --response_file "./responses/palo-spanish.json" --input_file "./responses/llava-1.5-7b-hf-Spanish.json"
# python3 evaluate.py --response_file "./responses/palo-indonesian.json" --input_file "./responses/llava-1.5-7b-hf-Indonesian.json"
# python3 evaluate.py --response_file "./responses/palo-kinyarwanda.json" --input_file "./responses/llava-1.5-7b-hf-Kinyarwanda.json"

# python3 evaluate.py --response_file "./responses/cambrian-korean.json" --input_file "./responses/llava-1.5-7b-hf-Korean.json"
# python3 evaluate.py --response_file "./responses/cambrian-chinese.json" --input_file "./responses/llava-1.5-7b-hf-Chinese.json"
# python3 evaluate.py --response_file "./responses/cambrian-english.json" --input_file "./responses/llava-1.5-7b-hf-English.json"
# python3 evaluate.py --response_file "./responses/cambrian-japanese.json" --input_file "./responses/llava-1.5-7b-hf-Japanese.json"
# python3 evaluate.py --response_file "./responses/cambrian-hindi.json" --input_file "./responses/llava-1.5-7b-hf-Hindi.json"
# python3 evaluate.py --response_file "./responses/cambrian-spanish.json" --input_file "./responses/llava-1.5-7b-hf-Spanish.json"
# python3 evaluate.py --response_file "./responses/cambrian-indonesian.json" --input_file "./responses/llava-1.5-7b-hf-Indonesian.json"
# python3 evaluate.py --response_file "./responses/cambrian-kinyarwanda.json" --input_file "./responses/llava-1.5-7b-hf-Kinyarwanda.json"

# python3 evaluate.py --response_file "./responses/llava-1.5-7b-hf-Korean.json" --input_file "./responses/llava-1.5-7b-hf-Korean.json"
# python3 evaluate.py --response_file "./responses/llava-1.5-7b-hf-Chinese.json" --input_file "./responses/llava-1.5-7b-hf-Chinese.json"
# python3 evaluate.py --response_file "./responses/llava-1.5-7b-hf-English.json" --input_file "./responses/llava-1.5-7b-hf-English.json"
# python3 evaluate.py --response_file "./responses/llava-1.5-7b-hf-Japanese.json" --input_file "./responses/llava-1.5-7b-hf-Japanese.json"
# python3 evaluate.py --response_file "./responses/llava-1.5-7b-hf-Hindi.json" --input_file "./responses/llava-1.5-7b-hf-Hindi.json"
# python3 evaluate.py --response_file "./responses/llava-1.5-7b-hf-Spanish.json" --input_file "./responses/llava-1.5-7b-hf-Spanish.json"
# python3 evaluate.py --response_file "./responses/llava-1.5-7b-hf-Indonesian.json" --input_file "./responses/llava-1.5-7b-hf-Indonesian.json"
# python3 evaluate.py --response_file "./responses/llava-1.5-7b-hf-Kinyarwanda.json" --input_file "./responses/llava-1.5-7b-hf-Kinyarwanda.json"

# python3 evaluate.py --response_file "./responses/llava-v1.6-vicuna-7b-hf-Korean.json" --input_file "./responses/llava-1.5-7b-hf-Korean.json"
# python3 evaluate.py --response_file "./responses/llava-v1.6-vicuna-7b-hf-Chinese.json" --input_file "./responses/llava-1.5-7b-hf-Chinese.json"
# python3 evaluate.py --response_file "./responses/llava-v1.6-vicuna-7b-hf-English.json" --input_file "./responses/llava-1.5-7b-hf-English.json"
# python3 evaluate.py --response_file "./responses/llava-v1.6-vicuna-7b-hf-Japanese.json" --input_file "./responses/llava-1.5-7b-hf-Japanese.json"
# python3 evaluate.py --response_file "./responses/llava-v1.6-vicuna-7b-hf-Hindi.json" --input_file "./responses/llava-1.5-7b-hf-Hindi.json"
# python3 evaluate.py --response_file "./responses/llava-v1.6-vicuna-7b-hf-Spanish.json" --input_file "./responses/llava-1.5-7b-hf-Spanish.json"
# python3 evaluate.py --response_file "./responses/llava-v1.6-vicuna-7b-hf-Indonesian.json" --input_file "./responses/llava-1.5-7b-hf-Indonesian.json"
# python3 evaluate.py --response_file "./responses/llava-v1.6-vicuna-7b-hf-Kinyarwanda.json" --input_file "./responses/llava-1.5-7b-hf-Kinyarwanda.json"

# python3 evaluate.py --response_file "./responses/mblip-bloomz-7b-Korean.json" --input_file "./responses/llava-1.5-7b-hf-Korean.json"
# python3 evaluate.py --response_file "./responses/mblip-bloomz-7b-Chinese.json" --input_file "./responses/llava-1.5-7b-hf-Chinese.json"
# python3 evaluate.py --response_file "./responses/mblip-bloomz-7b-English.json" --input_file "./responses/llava-1.5-7b-hf-English.json"
# python3 evaluate.py --response_file "./responses/mblip-bloomz-7b-Japanese.json" --input_file "./responses/llava-1.5-7b-hf-Japanese.json"
# python3 evaluate.py --response_file "./responses/mblip-bloomz-7b-Hindi.json" --input_file "./responses/llava-1.5-7b-hf-Hindi.json"
# python3 evaluate.py --response_file "./responses/mblip-bloomz-7b-Spanish.json" --input_file "./responses/llava-1.5-7b-hf-Spanish.json"
# python3 evaluate.py --response_file "./responses/mblip-bloomz-7b-Indonesian.json" --input_file "./responses/llava-1.5-7b-hf-Indonesian.json"
# python3 evaluate.py --response_file "./responses/mblip-bloomz-7b-Kinyarwanda.json" --input_file "./responses/llava-1.5-7b-hf-Kinyarwanda.json"

# python3 evaluate.py --response_file "./responses/mblip-mt0-xl-Korean.json" --input_file "./responses/llava-1.5-7b-hf-Korean.json"
# python3 evaluate.py --response_file "./responses/mblip-mt0-xl-Chinese.json" --input_file "./responses/llava-1.5-7b-hf-Chinese.json"
# python3 evaluate.py --response_file "./responses/mblip-mt0-xl-English.json" --input_file "./responses/llava-1.5-7b-hf-English.json"
# python3 evaluate.py --response_file "./responses/mblip-mt0-xl-Japanese.json" --input_file "./responses/llava-1.5-7b-hf-Japanese.json"
# python3 evaluate.py --response_file "./responses/mblip-mt0-xl-Hindi.json" --input_file "./responses/llava-1.5-7b-hf-Hindi.json"
# python3 evaluate.py --response_file "./responses/mblip-mt0-xl-Spanish.json" --input_file "./responses/llava-1.5-7b-hf-Spanish.json"
# python3 evaluate.py --response_file "./responses/mblip-mt0-xl-Indonesian.json" --input_file "./responses/llava-1.5-7b-hf-Indonesian.json"
# python3 evaluate.py --response_file "./responses/mblip-mt0-xl-Kinyarwanda.json" --input_file "./responses/llava-1.5-7b-hf-Kinyarwanda.json"

# python3 evaluate.py --response_file "./responses/paligemma-3b-mix-448-Korean.json" --input_file "./responses/llava-1.5-7b-hf-Korean.json"
# python3 evaluate.py --response_file "./responses/paligemma-3b-mix-448-Chinese.json" --input_file "./responses/llava-1.5-7b-hf-Chinese.json"
# python3 evaluate.py --response_file "./responses/paligemma-3b-mix-448-English.json" --input_file "./responses/llava-1.5-7b-hf-English.json"
# python3 evaluate.py --response_file "./responses/paligemma-3b-mix-448-Japanese.json" --input_file "./responses/llava-1.5-7b-hf-Japanese.json"
# python3 evaluate.py --response_file "./responses/paligemma-3b-mix-448-Hindi.json" --input_file "./responses/llava-1.5-7b-hf-Hindi.json"
# python3 evaluate.py --response_file "./responses/paligemma-3b-mix-448-Spanish.json" --input_file "./responses/llava-1.5-7b-hf-Spanish.json"
# python3 evaluate.py --response_file "./responses/paligemma-3b-mix-448-Indonesian.json" --input_file "./responses/llava-1.5-7b-hf-Indonesian.json"
# python3 evaluate.py --response_file "./responses/paligemma-3b-mix-448-Kinyarwanda.json" --input_file "./responses/llava-1.5-7b-hf-Kinyarwanda.json"

# python3 evaluate.py --response_file "./responses/Phi-3.5-vision-instruct-Korean.json" --input_file "./responses/llava-1.5-7b-hf-Korean.json"
# python3 evaluate.py --response_file "./responses/Phi-3.5-vision-instruct-Chinese.json" --input_file "./responses/llava-1.5-7b-hf-Chinese.json"
# python3 evaluate.py --response_file "./responses/Phi-3.5-vision-instruct-English.json" --input_file "./responses/llava-1.5-7b-hf-English.json"
# python3 evaluate.py --response_file "./responses/Phi-3.5-vision-instruct-Japanese.json" --input_file "./responses/llava-1.5-7b-hf-Japanese.json"
# python3 evaluate.py --response_file "./responses/Phi-3.5-vision-instruct-Hindi.json" --input_file "./responses/llava-1.5-7b-hf-Hindi.json"
# python3 evaluate.py --response_file "./responses/Phi-3.5-vision-instruct-Spanish.json" --input_file "./responses/llava-1.5-7b-hf-Spanish.json"
# python3 evaluate.py --response_file "./responses/Phi-3.5-vision-instruct-Indonesian.json" --input_file "./responses/llava-1.5-7b-hf-Indonesian.json"
# python3 evaluate.py --response_file "./responses/Phi-3.5-vision-instruct-Kinyarwanda.json" --input_file "./responses/llava-1.5-7b-hf-Kinyarwanda.json"

# python3 evaluate.py --response_file "./responses/gpt4o-Korean.json" --input_file "./responses/llava-1.5-7b-hf-Korean.json"
# python3 evaluate.py --response_file "./responses/gpt4o-Chinese.json" --input_file "./responses/llava-1.5-7b-hf-Chinese.json"
# python3 evaluate.py --response_file "./responses/gpt4o-English.json" --input_file "./responses/llava-1.5-7b-hf-English.json"
# python3 evaluate.py --response_file "./responses/gpt4o-Japanese.json" --input_file "./responses/llava-1.5-7b-hf-Japanese.json"
# python3 evaluate.py --response_file "./responses/gpt4o-Hindi.json" --input_file "./responses/llava-1.5-7b-hf-Hindi.json"
# python3 evaluate.py --response_file "./responses/gpt4o-Spanish.json" --input_file "./responses/llava-1.5-7b-hf-Spanish.json"
# python3 evaluate.py --response_file "./responses/gpt4o-Indonesian.json" --input_file "./responses/llava-1.5-7b-hf-Indonesian.json"

# python3 evaluate.py --response_file "./responses/gemini-1.5-pro-Korean.json" --input_file "./responses/llava-1.5-7b-hf-Korean.json"
# python3 evaluate.py --response_file "./responses/gemini-1.5-pro-Chinese.json" --input_file "./responses/llava-1.5-7b-hf-Chinese.json"
# python3 evaluate.py --response_file "./responses/gemini-1.5-pro-English.json" --input_file "./responses/llava-1.5-7b-hf-English.json"
# python3 evaluate.py --response_file "./responses/gemini-1.5-pro-Japanese.json" --input_file "./responses/llava-1.5-7b-hf-Japanese.json"
# python3 evaluate.py --response_file "./responses/gemini-1.5-pro-Hindi.json" --input_file "./responses/llava-1.5-7b-hf-Hindi.json"
# python3 evaluate.py --response_file "./responses/gemini-1.5-pro-Spanish.json" --input_file "./responses/llava-1.5-7b-hf-Spanish.json"
# python3 evaluate.py --response_file "./responses/gemini-1.5-pro-Indonesian.json" --input_file "./responses/llava-1.5-7b-hf-Indonesian.json"

# python3 evaluate.py --response_file "./responses/Molmo-7B-D-0924-Korean.json" --input_file "./responses/llava-1.5-7b-hf-Korean.json"
# python3 evaluate.py --response_file "./responses/Molmo-7B-D-0924-Chinese.json" --input_file "./responses/llava-1.5-7b-hf-Chinese.json"
python3 evaluate.py --response_file "./responses/Molmo-7B-D-0924-English.json" --input_file "./responses/llava-1.5-7b-hf-English.json"
python3 evaluate.py --response_file "./responses/Molmo-7B-D-0924-Japanese.json" --input_file "./responses/llava-1.5-7b-hf-Japanese.json"
python3 evaluate.py --response_file "./responses/Molmo-7B-D-0924-Hindi.json" --input_file "./responses/llava-1.5-7b-hf-Hindi.json"
python3 evaluate.py --response_file "./responses/Molmo-7B-D-0924-Spanish.json" --input_file "./responses/llava-1.5-7b-hf-Spanish.json"
python3 evaluate.py --response_file "./responses/Molmo-7B-D-0924-Indonesian.json" --input_file "./responses/llava-1.5-7b-hf-Indonesian.json"


python3 evaluate.py --response_file "./responses/llava-onevision-qwen2-7b-ov-chat-Korean.json" --input_file "./responses/llava-1.5-7b-hf-Korean.json"
python3 evaluate.py --response_file "./responses/llava-onevision-qwen2-7b-ov-chat-Chinese.json" --input_file "./responses/llava-1.5-7b-hf-Chinese.json"
python3 evaluate.py --response_file "./responses/llava-onevision-qwen2-7b-ov-chat-English.json" --input_file "./responses/llava-1.5-7b-hf-English.json"
python3 evaluate.py --response_file "./responses/llava-onevision-qwen2-7b-ov-chat-Japanese.json" --input_file "./responses/llava-1.5-7b-hf-Japanese.json"
python3 evaluate.py --response_file "./responses/llava-onevision-qwen2-7b-ov-chat-Hindi.json" --input_file "./responses/llava-1.5-7b-hf-Hindi.json"
python3 evaluate.py --response_file "./responses/llava-onevision-qwen2-7b-ov-chat-Spanish.json" --input_file "./responses/llava-1.5-7b-hf-Spanish.json"
python3 evaluate.py --response_file "./responses/llava-onevision-qwen2-7b-ov-chat-Indonesian.json" --input_file "./responses/llava-1.5-7b-hf-Indonesian.json"


python3 evaluate.py --response_file "./responses/Llama-3.2-11B-Vision-Instruct-Korean.json" --input_file "./responses/llava-1.5-7b-hf-Korean.json"
python3 evaluate.py --response_file "./responses/Llama-3.2-11B-Vision-Instruct-Chinese.json" --input_file "./responses/llava-1.5-7b-hf-Chinese.json"
python3 evaluate.py --response_file "./responses/Llama-3.2-11B-Vision-Instruct-English.json" --input_file "./responses/llava-1.5-7b-hf-English.json"
python3 evaluate.py --response_file "./responses/Llama-3.2-11B-Vision-Instruct-Japanese.json" --input_file "./responses/llava-1.5-7b-hf-Japanese.json"
python3 evaluate.py --response_file "./responses/Llama-3.2-11B-Vision-Instruct-Hindi.json" --input_file "./responses/llava-1.5-7b-hf-Hindi.json"
python3 evaluate.py --response_file "./responses/Llama-3.2-11B-Vision-Instruct-Spanish.json" --input_file "./responses/llava-1.5-7b-hf-Spanish.json"
python3 evaluate.py --response_file "./responses/Llama-3.2-11B-Vision-Instruct-Indonesian.json" --input_file "./responses/llava-1.5-7b-hf-Indonesian.json"