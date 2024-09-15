import argparse
import torch
import json
import os
import requests
from tqdm import tqdm





def prep_dataset(language):
    base_path = f"/home/seungonk/lmms-eval-mmllm/lmms_eval/tasks/xchat/{language}"
    eval_set = []
    
    i=0
    for task in os.listdir(base_path):
        task_path = os.path.join(base_path, task, "data.json")
        if os.path.exists(task_path):
            with open(task_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for index, item in enumerate(data):
                    prompt = f"{item['system_prompt']}\n\n{item['input']}"
                    image_path = os.path.join(base_path, task, f"{index}.jpg")
                    

                    eval_set.append({
                        "question_id": i+1,
                        "text":prompt,
                        "category":"",
                        "image":image_path
                    })
                    i+=1
                    
    return eval_set

def main(args):
    
    

    prompts = prep_dataset(args.language)

    # Save the prompts as a jsonl file
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for prompt in prompts:
            json.dump(prompt, f, ensure_ascii=False)
            f.write('\n')

    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate responses using vLLM")
    parser.add_argument("--language", type=str, required=True, help="Which language to evaluate")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the output file")
    

    args = parser.parse_args()
    main(args)