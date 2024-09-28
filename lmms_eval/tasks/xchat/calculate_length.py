import argparse
import json
import os
from transformers import AutoTokenizer
import numpy as np
import langdetect

def calculate_average_length(result_file):
    if "llava-v1.6" in result_file or "llava-1.5" in result_file:
        language = result_file.split("hf-")[1].split("-results")[0]
    else:
        language = result_file.split("responses/")[1].split("_prompts")[0]
    
    # Load the Qwen2 tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct",cache_dir="/data/tir/projects/tir7/user_data/seungonk/huggingface")

    # Read the result file
    with open(result_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Calculate lengths
    lengths = []
    non_target_language_count = 0
    total_responses = len(data)
    
    language_codes = {
        "Korean": "ko",
        "Chinese": "zh-cn",
        "English": "en",
        "Japanese": "ja",
        "Spanish": "es",
        "Indonesian": "id",
        "Kinyarwanda": "rw",
        "Hindi": "hi",
        "korean": "ko",
        "chinese": "zh-cn",
        "english": "en",
        "japanese": "ja",
        "spanish": "es",
        "indonesian": "id",
        "kinyarwanda": "rw",
        "hindi": "hi"
    }
    
    target_lang_code = language_codes[language]

    for item in data:
        response = item['response']
        if "ASSISTANT:" in response:
            response = response.split("ASSISTANT:")[1]
        elif "ER:" in response:
            response = response.split("\n\n",1)[1]
        
        try:
            detected_lang = langdetect.detect(response)

            if detected_lang == target_lang_code:
                tokens = tokenizer.encode(response)
                lengths.append(len(tokens))
            else:
                non_target_language_count += 1
        except langdetect.lang_detect_exception.LangDetectException:
            non_target_language_count += 1

    # Calculate average length
    if lengths:
        average_length = np.mean(lengths)
        print(result_file)
        print(f"Average response length: {average_length:.2f} tokens")
        print(f"Number of responses not in target language: {non_target_language_count}")
        print()
    else:
        print(result_file)
        print("No responses in target language found.")
        print(f"Number of responses not in target language: {non_target_language_count}")
        print()

    if (1 - non_target_language_count/total_responses) < 0.3:
        print(f"WARNING: Less than 30% of responses are in the target language ({language}).")
        print()

def process_directory(directory):
    for filename in os.listdir(directory):
        if filename.endswith("-results.json"):
            file_path = os.path.join(directory, filename)
            calculate_average_length(file_path)

def main():
    parser = argparse.ArgumentParser(description="Calculate average response length using Qwen2 tokenizer")
    parser.add_argument("--directory", type=str, required=True, help="Path to the directory containing result files")
    args = parser.parse_args()

    process_directory(args.directory)

if __name__ == "__main__":
    main()