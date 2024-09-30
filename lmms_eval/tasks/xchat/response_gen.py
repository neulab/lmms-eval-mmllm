import argparse
import torch
import json
import os
from PIL import Image
import requests
from tqdm import tqdm
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, LlavaNextForConditionalGeneration, PaliGemmaForConditionalGeneration, Blip2ForConditionalGeneration, Blip2Processor, AutoModelForCausalLM


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
    
    # processor.tokenizer.add_tokens(["<image>", "<pad>"], special_tokens=True)

    prompts, images, orig_data = prep_dataset(args.language)
    
    if "1.5" in args.model_name:
        model = LlavaForConditionalGeneration.from_pretrained(
            args.model_name, 
            torch_dtype=torch.float16,
            cache_dir="/data/tir/projects/tir7/user_data/seungonk/huggingface"
        ).to(0)
        model.resize_token_embeddings(len(processor.tokenizer))
        processor = AutoProcessor.from_pretrained(args.model_name)
    elif "3.5" in args.model_name:
        model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3.5-vision-instruct", torch_dtype=torch.bfloat16, trust_remote_code=True,cache_dir="/data/tir/projects/tir7/user_data/seungonk/huggingface").to(0)
        processor = AutoProcessor.from_pretrained(args.model_name,trust_remote_code=True,num_crops=4)
    elif "mix-448" in args.model_name:
        model = PaliGemmaForConditionalGeneration.from_pretrained("google/paligemma-3b-mix-448",trust_remote_code=True,torch_dtype=torch.bfloat16,cache_dir="/data/tir/projects/tir7/user_data/seungonk/huggingface").to(0)
        processor = AutoProcessor.from_pretrained(args.model_name,trust_remote_code=True)
    elif "1.6" in args.model_name:
        model = LlavaNextForConditionalGeneration.from_pretrained(
            args.model_name, 
            torch_dtype=torch.float16,
            cache_dir="/data/tir/projects/tir7/user_data/seungonk/huggingface"
        ).to(0)
        model.resize_token_embeddings(len(processor.tokenizer))
        processor = AutoProcessor.from_pretrained(args.model_name)
    elif "mblip-mt0-xl" in args.model_name:
        processor = Blip2Processor.from_pretrained("Gregor/mblip-mt0-xl")
        model = Blip2ForConditionalGeneration.from_pretrained("Gregor/mblip-mt0-xl", torch_dtype=torch.bfloat16, cache_dir="/data/tir/projects/tir7/user_data/seungonk/huggingface").to(0)
    elif "mblip-bloomz-7b" in args.model_name:
        processor = Blip2Processor.from_pretrained("Gregor/mblip-bloomz-7b")
        model = Blip2ForConditionalGeneration.from_pretrained("Gregor/mblip-bloomz-7b", torch_dtype=torch.bfloat16, cache_dir="/data/tir/projects/tir7/user_data/seungonk/huggingface").to(0)
    else:
        raise ValueError(f"Model name {args.model_name} is not supported")

    results = []
    for p,i in tqdm(zip(prompts,images),total=len(images)):
        if "1.5" in args.model_name or "1.6" in args.model_name:
            p = processor.apply_chat_template([p], add_generation_prompt=True)
            model_inputs = processor(images=i, text=p, return_tensors='pt').to("cuda", torch.float16)
            output = model.generate(**model_inputs, max_new_tokens=1024, min_new_tokens=32, temperature=1.0, top_p=0.9, do_sample=True)
            output = output[0]
            result = processor.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        elif "3.5" in args.model_name:
            p = f"<|user|>\n<|image_1|>\n{p['content'][1]['text']}<|end|>\n<|assistant|>\n"
            model_inputs = processor(images=i, text=p, return_tensors='pt').to("cuda", torch.bfloat16)
            output = model.generate(**model_inputs, max_new_tokens=1024, temperature=1.0, top_p=0.9, do_sample=True, eos_token_id=processor.tokenizer.eos_token_id)
            output = output[:, model_inputs['input_ids'].shape[1]:]
            result = processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        elif "mblip" in args.model_name:
            p = "Question: "+ p['content'][1]['text'].split("\n\n")[1] + f"\nAnswer in {args.language}: "
            model_inputs = processor(images=i, text=p, return_tensors='pt').to("cuda", torch.bfloat16)
            output = model.generate(**model_inputs, max_new_tokens=1024, min_new_tokens=32, repetition_penalty=1.1, top_p=0.9, do_sample=True)
            output = output[0]
            result = processor.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        elif "paligemma" in args.model_name:
            p = "Question: "+ p['content'][1]['text'].split("\n\n")[1] + f"\nAnswer in {args.language}: "
            model_inputs = processor(images=i, text=p, return_tensors='pt').to("cuda", torch.bfloat16)
            output = model.generate(**model_inputs, max_new_tokens=1024, repetition_penalty=1.1, top_p=0.9, temperature=1.0, do_sample=True)
            output = output[0]
            result = processor.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        
        if "Answer in" in result:
            result = result.split(f"Answer in {args.language}")[1]
        elif "ASSISTANT:" in result:
            result = result.split("ASSISTANT:")[1]
        elif "<|assistant|>" in result:
            result = result.split("<|assistant|>")[1]
        
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
    parser.add_argument("--gpu_num", type=int, default=1, help="Number of GPUs to use")

    args = parser.parse_args()
    main(args)