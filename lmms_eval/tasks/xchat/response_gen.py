import argparse
import torch
import json
import os
from PIL import Image
import requests
from tqdm import tqdm
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, LlavaNextForConditionalGeneration

def load_and_process_image(image_path):
    image = Image.open(image_path)
    return image

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
                    image = load_and_process_image(image_path)

                    eval_set.append({
                        "role": "user",
                        "content":[
                            {"type":"image"},
                            {"type":"text","text":prompt}
                        ]
                    })
                    images.append(image)
                    orig_data.append(item)
                    
    
    return eval_set, images, orig_data

def main(args):
    
    # if "llava-hf" in args.model_name:
    #     processor = AutoProcessor.from_pretrained(args.model_name)
    # else:
    #     processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    processor = AutoProcessor.from_pretrained(args.model_name)
    processor.tokenizer.add_tokens(["<image>", "<pad>"], special_tokens=True)

    prompts, images, orig_data = prep_dataset(args.language)
    
    if "1.5" in args.model_name:
        model = LlavaForConditionalGeneration.from_pretrained(
            args.model_name, 
            torch_dtype=torch.float16
        ).to(0)
        model.resize_token_embeddings(len(processor.tokenizer))
    else:
        model = LlavaNextForConditionalGeneration.from_pretrained(
            args.model_name, 
            torch_dtype=torch.float16
        ).to(0)
        model.resize_token_embeddings(len(processor.tokenizer))

    results = []
    for p,i in tqdm(zip(prompts,images),total=len(images)):
        p = processor.apply_chat_template([p], add_generation_prompt=True)
        
        model_inputs = processor(images=i, text=p, return_tensors='pt').to(0, torch.float16)
        
        output = model.generate(**model_inputs, max_new_tokens=1024, temperature=1.0, top_p=0.9, do_sample=True)
        results.append(processor.decode(output[0][2:], skip_special_tokens=True))

    for inst,r in zip(orig_data,results):
        inst['response'] = r

    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(orig_data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate responses using vLLM")
    parser.add_argument("--model_name", type=str, required=True, help="huggingface checkpoint")
    parser.add_argument("--language", type=str, required=True, help="Which language to evaluate")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the output file")
    parser.add_argument("--gpu_num", type=int, default=1, help="Number of GPUs to use")

    args = parser.parse_args()
    main(args)