#!/bin/sh
#SBATCH --job-name=cvqa-llava-1.5
#SBATCH --output /home/skhanuja/lmms-eval-mmllm/lmms_eval/tasks/scratch_cvqa-llava-1.5
#SBATCH --nodes=1
#SBATCH --mem=50GB
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:A6000:1
#SBATCH --time 2-00:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=skhanuja@andrew.cmu.edu

# activate conda
source ~/miniconda3/etc/profile.d/conda.sh 
conda activate lmms-eval


echo ${HOSTNAME}
export HF_HOME=/scratch/${USER}/cache

export HF_TOKEN="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"

#languages = ['ar', 'bn', 'cs', 'da', 'de', 'el', 'es', 'fa', 'fi', 'fil', 'fr', 'hi', 'hr', 'hu', 'id', 'it', 'he', 'ja', 'ko', 'mi', 'nl', 'no', 'pl', 'pt', 'quz', 'ro', 'ru', 'sv', 'sw', 'te', 'th', 'tr', 'vi', 'zh']
#languages_full = ['Arabic', 'Bengali', 'Czech', 'Danish', 'German', 'Greek', 'Spanish', 'Persian', 'Finnish', 'Filipino', 'French', 'Hindi', 'Croatian', 'Hungarian', 'Indonesian', 'Italian', 'Hebrew', 'Japanese', 'Korean', 'Maori', 'Dutch', 'Norwegian', 'Polish', 'Portuguese', 'Quechua', 'Romanian', 'Russian', 'Swedish', 'Swahili', 'Telugu', 'Thai', 'Turkish', 'Vietnamese', 'Chinese']
tasks="cvqa_full"
accelerate launch --main_process_port 0 --num_processes=1 -m lmms_eval \
--model llava \
--model_args pretrained="liuhaotian/llava-v1.5-7b" \
--tasks cvqa_test  \
--batch_size 32 \
--log_samples \
--log_samples_suffix llava_v1.5_cvqa \
--output_path ./logs/