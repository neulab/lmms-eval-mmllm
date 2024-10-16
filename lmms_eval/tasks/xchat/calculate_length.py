import argparse
import json
import os
from transformers import AutoTokenizer
import numpy as np
import langdetect

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

def calculate_average_length(result_file):
    for lc in language_codes.keys():
        if lc in result_file:
            language = lc
            break
        
    
    # Load the Qwen2 tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")

    # Read the result file
    with open(result_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Calculate lengths
    lengths = []
    non_target_language_count = 0
    total_responses = len(data)
    
    
    
    target_lang_code = language_codes[language]

    original_scores = []
    adjusted_scores = []

    for item in data:
        response = item['response']
        if response is not None:
            if "ASSISTANT:" in response:
                response = response.split("ASSISTANT:")[1]
            elif "ER:" in response:
                response = response.split("\n\n",1)[1]
        else:
            response = " "
        
        original_score = item.get('score', 0)
        original_scores.append(original_score)

        try:
            detected_lang = langdetect.detect(response)

            if detected_lang == target_lang_code:
                tokens = tokenizer.encode(response)
                lengths.append(len(tokens))
                adjusted_scores.append(original_score)
            else:
                non_target_language_count += 1
                adjusted_scores.append(1)
        except langdetect.lang_detect_exception.LangDetectException:
            non_target_language_count += 1
            adjusted_scores.append(1)

    # Calculate average length
    if lengths:
        average_length = np.mean(lengths)
        print(result_file)
        print(f"Average response length: {average_length:.2f} tokens")
        print(f"Number of responses not in target language: {non_target_language_count}")
        print(f"Original scores: {sum(original_scores)/len(original_scores)}")
        print(f"Adjusted scores: {sum(adjusted_scores)/len(adjusted_scores)}")
        print()
    else:
        print(result_file)
        print("No responses in target language found.")
        print(f"Number of responses not in target language: {non_target_language_count}")
        print(f"Original scores: {sum(original_scores)/len(original_scores)}")
        print(f"Adjusted scores: {sum(adjusted_scores)/len(adjusted_scores)}")
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