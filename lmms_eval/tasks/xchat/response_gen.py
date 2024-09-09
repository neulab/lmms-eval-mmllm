import argparse
import json
from typing import List, Dict
from vllm import LLM, SamplingParams
from conversation import get_conv_template
import os

import torch
from PIL import Image
from torchvision import transforms

def load_and_process_image(image_path, size=(336, 336)):
    if os.path.exists(image_path):
        image = Image.open(image_path)
        transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
        ])
        return transform(image).unsqueeze(0)  # Add batch dimension
    else:
        return torch.zeros((1, 3, *size))  # Placeholder for missing images

def process_dialogue(inst,system_message):
    return "<image>" * 576+f"\n{system_message}"+f"\nUSER: {inst}\nASSISTANT: "
    

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
                for index,item in enumerate(data):
                    eval_set.append(process_dialogue(item["input"],item['system_prompt']))
                    orig_data.append(item)
                    image_path = os.path.join(base_path, task, f"{index}.jpg")
                    images.append(load_and_process_image(image_path))

    return eval_set, images, orig_data
    

def main(args):
    eval_suite,orig_data,images = prep_dataset(args.language)

    sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=2048, stop=["<|eot_id|>","<|end_of_text|>","# Test the function","<pad>"])

    llm = LLM(model=args.model_name,tensor_parallel_size=args.gpu_num,gpu_memory_utilization=0.95,download_dir="/data/tir/projects/tir7/user_data/seungonk/huggingface", image_input_type="pixel_values", image_token_id=32000, image_input_shape="1,3,336,336", image_feature_size=576)
    outputs = llm.generate(prompts=eval_suite, sampling_params=sampling_params, multi_modal_data=MultiModalData(type=MultiModalData.Type.IMAGE, data=images))

    results = []
    for output,inst in zip(outputs,orig_data):
        x = inst
        x['response'] = output.outputs[0].text
        results.append(x)

    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate responses using vLLM")
    parser.add_argument("--model_name", type=str, required=True, help="huggingface checkpoint")
    parser.add_argument("--language", type=str, required=True, help="Which language to evaluate")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the output file")
    parser.add_argument("--gpu_num", type=int, default=1, help="Number of GPUs to use")

    args = parser.parse_args()
    main(args)