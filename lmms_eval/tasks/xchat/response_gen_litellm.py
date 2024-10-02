import argparse
import torch
import json
import os
from PIL import Image
import requests
from tqdm import tqdm
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, LlavaNextForConditionalGeneration, PaliGemmaForConditionalGeneration, Blip2ForConditionalGeneration, Blip2Processor, AutoModelForCausalLM
import base64
import litellm

litellm.verbose =True

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def prep_dataset(language):
    base_path = f"./{language}"
    eval_set = []
    orig_data = []
    images = []

    for task in os.listdir(base_path):
        task_path = os.path.join(base_path, task, "data.json")
        if os.path.exists(task_path):
            with open(task_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for index, item in enumerate(data):
                    prompt = f"{item['system_prompt']}\n\n{item['input']}"
                    image_path = os.path.join(base_path, task, f"{index}.jpg")

                    eval_set.append({
                        "system_message": item['system_prompt'],
                        "prompt": item['input'],
                        "image": encode_image(image_path)
                    })
                    
                    orig_data.append(item)
                    
    
    return eval_set, orig_data

def main(args):

    eval_data, orig_data = prep_dataset(args.language)
    
    results = []
    for e in tqdm(eval_data,total=len(eval_data)):
        
        response = litellm.completion(
                model=args.model_name,
                base_url="https://cmu.litellm.ai",
                api_key="sk-nvPQWSrFY02nf-1ZtPGIFw",
                messages=[{
                    'role': 'system',
                    'content': e['system_message']
                }, {
                    'role': 'user',
                    'content': [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{e['image']}"}}
                    ]
                }, {
                    'role': 'user',
                    'content': e['prompt']
                }],
                max_tokens=1024,
                temperature=1.0,
                top_p=0.9
            )
        result = response['choices'][0]['message']['content']
        results.append(result)

    for inst,r in zip(orig_data,results):
        inst['response'] = r

    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(orig_data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate responses using vLLM")
    parser.add_argument("--model_name", type=str, required=True, help="huggingface checkpoint")
    parser.add_argument("--language", type=str, required=True, help="Which language to evaluate")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the output file")

    args = parser.parse_args()
    main(args)