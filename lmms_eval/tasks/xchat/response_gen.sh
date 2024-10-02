#!/bin/bash
#SBATCH --job-name=seungone-xchat
#SBATCH --output=xchat.out
#SBATCH --error=xchat.err
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2:00:00
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
# python3 response_gen.py --model_name "Gregor/mblip-mt0-xl" --language "English" --output_file "./responses/mblip-mt0-xl-English.json" --gpu_num 1
# python3 response_gen.py --model_name "Gregor/mblip-bloomz-7b" --language "English" --output_file "./responses/mblip-bloomz-7b-English.json" --gpu_num 1
# python3 response_gen.py --model_name "google/paligemma-3b-mix-448" --language "English" --output_file "./responses/paligemma-3b-mix-448-English.json" --gpu_num 1
# python3 response_gen.py --model_name "microsoft/Phi-3.5-vision-instruct" --language "English" --output_file "./responses/Phi-3.5-vision-instruct-English.json" --gpu_num 1
python3 response_gen_litellm.py --model_name "openai/neulab/gpt-4o-2024-08-06" --language "English" --output_file "./responses/gpt4o-English.json"
python3 response_gen_litellm.py --model_name "openai/neulab/gemini/gemini-1.5-pro" --language "English" --output_file "./responses/gemini-1.5-pro-English.json"

# Chinese
# python3 response_gen.py --model_name "llava-hf/llava-1.5-7b-hf" --language "Chinese" --output_file "./responses/llava-1.5-7b-hf-Chinese.json" --gpu_num 1
# python3 response_gen.py --model_name "llava-hf/llava-v1.6-vicuna-7b-hf" --language "Chinese" --output_file "./responses/llava-v1.6-vicuna-7b-hf-Chinese.json" --gpu_num 1
# python3 response_gen.py --model_name "Gregor/mblip-mt0-xl" --language "Chinese" --output_file "./responses/mblip-mt0-xl-Chinese.json" --gpu_num 1
# python3 response_gen.py --model_name "Gregor/mblip-bloomz-7b" --language "Chinese" --output_file "./responses/mblip-bloomz-7b-Chinese.json" --gpu_num 1
# python3 response_gen.py --model_name "google/paligemma-3b-mix-448" --language "Chinese" --output_file "./responses/paligemma-3b-mix-448-Chinese.json" --gpu_num 1
# python3 response_gen.py --model_name "microsoft/Phi-3.5-vision-instruct" --language "Chinese" --output_file "./responses/Phi-3.5-vision-instruct-Chinese.json" --gpu_num 1
python3 response_gen_litellm.py --model_name "openai/neulab/gpt-4o-2024-08-06" --language "Chinese" --output_file "./responses/gpt4o-Chinese.json"
python3 response_gen_litellm.py --model_name "openai/neulab/gemini/gemini-1.5-pro" --language "Chinese" --output_file "./responses/gemini-1.5-pro-Chinese.json"

# Hindi
# python3 response_gen.py --model_name "llava-hf/llava-1.5-7b-hf" --language "Hindi" --output_file "./responses/llava-1.5-7b-hf-Hindi.json" --gpu_num 1
# python3 response_gen.py --model_name "llava-hf/llava-v1.6-vicuna-7b-hf" --language "Hindi" --output_file "./responses/llava-v1.6-vicuna-7b-hf-Hindi.json" --gpu_num 1
# python3 response_gen.py --model_name "Gregor/mblip-mt0-xl" --language "Hindi" --output_file "./responses/mblip-mt0-xl-Hindi.json" --gpu_num 1
# python3 response_gen.py --model_name "Gregor/mblip-bloomz-7b" --language "Hindi" --output_file "./responses/mblip-bloomz-7b-Hindi.json" --gpu_num 1
# python3 response_gen.py --model_name "google/paligemma-3b-mix-448" --language "Hindi" --output_file "./responses/paligemma-3b-mix-448-Hindi.json" --gpu_num 1
# python3 response_gen.py --model_name "microsoft/Phi-3.5-vision-instruct" --language "Hindi" --output_file "./responses/Phi-3.5-vision-instruct-Hindi.json" --gpu_num 1
python3 response_gen_litellm.py --model_name "openai/neulab/gpt-4o-2024-08-06" --language "Hindi" --output_file "./responses/gpt4o-Hindi.json"
python3 response_gen_litellm.py --model_name "openai/neulab/gemini/gemini-1.5-pro" --language "Hindi" --output_file "./responses/gemini-1.5-pro-Hindi.json"

# Indonesian
# python3 response_gen.py --model_name "llava-hf/llava-1.5-7b-hf" --language "Indonesian" --output_file "./responses/llava-1.5-7b-hf-Indonesian.json" --gpu_num 1
# python3 response_gen.py --model_name "llava-hf/llava-v1.6-vicuna-7b-hf" --language "Indonesian" --output_file "./responses/llava-v1.6-vicuna-7b-hf-Indonesian.json" --gpu_num 1
# python3 response_gen.py --model_name "Gregor/mblip-mt0-xl" --language "Indonesian" --output_file "./responses/mblip-mt0-xl-Indonesian.json" --gpu_num 1
# python3 response_gen.py --model_name "Gregor/mblip-bloomz-7b" --language "Indonesian" --output_file "./responses/mblip-bloomz-7b-Indonesian.json" --gpu_num 1
# python3 response_gen.py --model_name "google/paligemma-3b-mix-448" --language "Indonesian" --output_file "./responses/paligemma-3b-mix-448-Indonesian.json" --gpu_num 1
# python3 response_gen.py --model_name "microsoft/Phi-3.5-vision-instruct" --language "Indonesian" --output_file "./responses/Phi-3.5-vision-instruct-Indonesian.json" --gpu_num 1
python3 response_gen_litellm.py --model_name "openai/neulab/gpt-4o-2024-08-06" --language "Indonesian" --output_file "./responses/gpt4o-Indonesian.json"
python3 response_gen_litellm.py --model_name "openai/neulab/gemini/gemini-1.5-pro" --language "Indonesian" --output_file "./responses/gemini-1.5-pro-Indonesian.json"

# Japanese
# python3 response_gen.py --model_name "llava-hf/llava-1.5-7b-hf" --language "Japanese" --output_file "./responses/llava-1.5-7b-hf-Japanese.json" --gpu_num 1
# python3 response_gen.py --model_name "llava-hf/llava-v1.6-vicuna-7b-hf" --language "Japanese" --output_file "./responses/llava-v1.6-vicuna-7b-hf-Japanese.json" --gpu_num 1
# python3 response_gen.py --model_name "Gregor/mblip-mt0-xl" --language "Japanese" --output_file "./responses/mblip-mt0-xl-Japanese.json" --gpu_num 1
# python3 response_gen.py --model_name "Gregor/mblip-bloomz-7b" --language "Japanese" --output_file "./responses/mblip-bloomz-7b-Japanese.json" --gpu_num 1
# python3 response_gen.py --model_name "google/paligemma-3b-mix-448" --language "Japanese" --output_file "./responses/paligemma-3b-mix-448-Japanese.json" --gpu_num 1
# python3 response_gen.py --model_name "microsoft/Phi-3.5-vision-instruct" --language "Japanese" --output_file "./responses/Phi-3.5-vision-instruct-Japanese.json" --gpu_num 1
python3 response_gen_litellm.py --model_name "openai/neulab/gpt-4o-2024-08-06" --language "Japanese" --output_file "./responses/gpt4o-Japanese.json"
python3 response_gen_litellm.py --model_name "openai/neulab/gemini/gemini-1.5-pro" --language "Japanese" --output_file "./responses/gemini-1.5-pro-Japanese.json"

# Kinyarwanda
# python3 response_gen.py --model_name "llava-hf/llava-1.5-7b-hf" --language "Kinyarwanda" --output_file "./responses/llava-1.5-7b-hf-Kinyarwanda.json" --gpu_num 1
# python3 response_gen.py --model_name "llava-hf/llava-v1.6-vicuna-7b-hf" --language "Kinyarwanda" --output_file "./responses/llava-v1.6-vicuna-7b-hf-Kinyarwanda.json" --gpu_num 1
# python3 response_gen.py --model_name "Gregor/mblip-mt0-xl" --language "Kinyarwanda" --output_file "./responses/mblip-mt0-xl-Kinyarwanda.json" --gpu_num 1
# python3 response_gen.py --model_name "Gregor/mblip-bloomz-7b" --language "Kinyarwanda" --output_file "./responses/mblip-bloomz-7b-Kinyarwanda.json" --gpu_num 1
# python3 response_gen.py --model_name "google/paligemma-3b-mix-448" --language "Kinyarwanda" --output_file "./responses/paligemma-3b-mix-448-Kinyarwanda.json" --gpu_num 1
# python3 response_gen.py --model_name "microsoft/Phi-3.5-vision-instruct" --language "Kinyarwanda" --output_file "./responses/Phi-3.5-vision-instruct-Kinyarwanda.json" --gpu_num 1
# python3 response_gen_litellm.py --model_name "openai/neulab/gpt-4o-2024-08-06" --language "Kinyarwanda" --output_file "./responses/gpt4o-Kinyarwanda.json"
# python3 response_gen_litellm.py --model_name "openai/neulab/gemini/gemini-1.5-pro" --language "Kinyarwanda" --output_file "./responses/gemini-1.5-pro-Kinyarwanda.json"

# Korean
# python3 response_gen.py --model_name "llava-hf/llava-1.5-7b-hf" --language "Korean" --output_file "./responses/llava-1.5-7b-hf-Korean.json" --gpu_num 1
# python3 response_gen.py --model_name "llava-hf/llava-v1.6-vicuna-7b-hf" --language "Korean" --output_file "./responses/llava-v1.6-vicuna-7b-hf-Korean.json" --gpu_num 1
# python3 response_gen.py --model_name "Gregor/mblip-mt0-xl" --language "Korean" --output_file "./responses/mblip-mt0-xl-Korean.json" --gpu_num 1
# python3 response_gen.py --model_name "Gregor/mblip-bloomz-7b" --language "Korean" --output_file "./responses/mblip-bloomz-7b-Korean.json" --gpu_num 1
# python3 response_gen.py --model_name "google/paligemma-3b-mix-448" --language "Korean" --output_file "./responses/paligemma-3b-mix-448-Korean.json" --gpu_num 1
# python3 response_gen.py --model_name "microsoft/Phi-3.5-vision-instruct" --language "Korean" --output_file "./responses/Phi-3.5-vision-instruct-Korean.json" --gpu_num 1
python3 response_gen_litellm.py --model_name "openai/neulab/gpt-4o-2024-08-06" --language "Korean" --output_file "./responses/gpt4o-Korean.json"
python3 response_gen_litellm.py --model_name "openai/neulab/gemini/gemini-1.5-pro" --language "Korean" --output_file "./responses/gemini-1.5-pro-Korean.json"

# Spanish
# python3 response_gen.py --model_name "llava-hf/llava-1.5-7b-hf" --language "Spanish" --output_file "./responses/llava-1.5-7b-hf-Spanish.json" --gpu_num 1
# python3 response_gen.py --model_name "llava-hf/llava-v1.6-vicuna-7b-hf" --language "Spanish" --output_file "./responses/llava-v1.6-vicuna-7b-hf-Spanish.json" --gpu_num 1
# python3 response_gen.py --model_name "Gregor/mblip-mt0-xl" --language "Spanish" --output_file "./responses/mblip-mt0-xl-Spanish.json" --gpu_num 1
# python3 response_gen.py --model_name "Gregor/mblip-bloomz-7b" --language "Spanish" --output_file "./responses/mblip-bloomz-7b-Spanish.json" --gpu_num 1
# python3 response_gen.py --model_name "google/paligemma-3b-mix-448" --language "Spanish" --output_file "./responses/paligemma-3b-mix-448-Spanish.json" --gpu_num 1
# python3 response_gen.py --model_name "microsoft/Phi-3.5-vision-instruct" --language "Spanish" --output_file "./responses/Phi-3.5-vision-instruct-Spanish.json" --gpu_num 1
python3 response_gen_litellm.py --model_name "openai/neulab/gpt-4o-2024-08-06" --language "Spanish" --output_file "./responses/gpt4o-Spanish.json"
python3 response_gen_litellm.py --model_name "openai/neulab/gemini/gemini-1.5-pro" --language "Spanish" --output_file "./responses/gemini-1.5-pro-Spanish.json"