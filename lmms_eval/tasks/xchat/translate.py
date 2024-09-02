import argparse
import json
import os
import time
import concurrent.futures

# import google.generativeai as genai
import openai
import tqdm
import tiktoken
from copy import deepcopy
import random
import copy

import shutil

# with open("./key.txt",'r') as f:
#     genai.configure(api_key=f.read())

# model = genai.GenerativeModel("gemini-1.5-pro")

openai.api_key = ""
with open("./key.txt",'r') as f:
    os.environ["OPENAI_API_KEY"] = f.read()

client = openai.OpenAI()

prompt_template = """Translate the following SENTENCES from English to <target_language_>.
DO NOT WRITE ANY GREETING MESSAGES; just the requested SENTENCES.
Write [END] after you are done.

SENTENCES: <sentence_>"""

language_list = ["Chinese", "Japanese", "Korean", "Hindi", "Indonesia", "Kinyarwanda", "Spanish"]

def traverse_and_process_data(root_dir):
    final_results = {}
    for subdir, _, files in os.walk(root_dir):
        rel_path = os.path.relpath(subdir, root_dir)
        subdir_name = rel_path.split(os.path.sep)[0] if rel_path != '.' else ''
        
        for file in files:
            if file == 'data.json':
                file_path = os.path.join(subdir, file)
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    # Process the data here
                    final_results[subdir_name] = data
    return final_results


def translate(orig_sentence, language, task_name, inst_idx, type_, orig_instance):
    input_ = prompt_template.replace("<target_language_>",language).replace("<sentence_>",orig_sentence)
    while True:
        try:
            ### GEMINI ###
            # response = model.generate_content(
            #     input_,
            #     generation_config=genai.types.GenerationConfig(
            #         candidate_count=1,
            #         max_output_tokens=2048,
            #         temperature=0.3,
            #         stop_sequences="[END]"
            #     )
            # )
            # output_ = response.text.split("[END]")[0]
            ###

            response = client.chat.completions.create(
                model="gpt-4o-2024-08-06",
                messages=[{
                    'role': 'system',
                    'content': 'You are an AI model assigned for a translation task.'
                }, {
                    'role': 'user',
                    'content': input_,
                }],
                max_tokens=2048,
                temperature=0.3,
                top_p=0.9,
                stop="[END]"
            )
            output_ = response.choices[0].message.content.split("[END]")[0]

            return {"output": output_, "language": language, "task_name": task_name, "instance_idx": inst_idx, "type": type_, "orig_instance": orig_instance}

        except Exception as e:
            print('[ERROR]', e)
            time.sleep(5)


if __name__ == '__main__':    

    english_files = traverse_and_process_data("./English")
    
        
    tmp_results1=[]
    tmp_results2=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        futures1 = []
        futures2 = []
        for l in language_list:
            for task_name in english_files.keys():
                for d in english_files[task_name]:
                    future1 = executor.submit(translate, d['input'], l, task_name, d['instance_idx'], "input",d)
                    future2 = executor.submit(translate, d['reference_answer'], l, task_name, d['instance_idx'], "reference_answer",d)
                    futures1.append(future1)
                    futures2.append(future2)

        for future in tqdm.tqdm(concurrent.futures.as_completed(futures1), total=len(futures1)):
            tmp_results1.append(future.result())
        for future in tqdm.tqdm(concurrent.futures.as_completed(futures2), total=len(futures2)):
            tmp_results2.append(future.result())

    final_results = {}
    for l in language_list:
        final_results[l] = {}
    for inst in tmp_results1:
        x = copy.deepcopy(inst['orig_instance'])

        x['input'] = inst['output']

        if x['task'] not in final_results[inst['language']]:
            final_results[inst['language']][inst['task_name']] = []

        final_results[inst['language']][inst['task_name']].append(x)

    for inst in tmp_results2:
        for d in final_results[inst['language']][inst['task_name']]:
            if d['instance_idx'] == inst['instance_idx']:
                d['reference_answer'] = inst['output']

    for language in final_results:
        for task in final_results[language]:
            final_results[language][task] = sorted(final_results[language][task], key=lambda x: x['instance_idx'])
            language_task_dir = os.path.join(os.getcwd(), language, task)
            os.makedirs(language_task_dir, exist_ok=True)
            
            
            with open(os.path.join(language_task_dir, "data.json"), 'w', encoding='utf-8') as f:
                json.dump(final_results[language][task], f, ensure_ascii=False, indent=4)
            
            # Copy image files from English directory to the corresponding language directory
            english_task_dir = os.path.join(os.getcwd(), "English", task)
            for i in range(5):  # Assuming there are always 5 images (0.jpg to 4.jpg)
                src_image = os.path.join(english_task_dir, f"{i}.jpg")
                if os.path.exists(src_image):
                    dst_image = os.path.join(language_task_dir, f"{i}.jpg")
                    shutil.copy2(src_image, dst_image)