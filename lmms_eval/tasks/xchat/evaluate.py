# Absolute Grading: Outputs score of 1 to 5

import os
import json
from prometheus_eval.litellm import AsyncLiteLLM
from prometheus_eval.vllm import VLLM
from prometheus_eval import PrometheusEval
from prometheus_eval.prompts import ABSOLUTE_PROMPT, SCORE_RUBRIC_TEMPLATE
from tqdm import tqdm
import argparse
import json
import os


def prep_dataset(language):
    base_path = f"./{language}"
    orig_data = []

    for task in os.listdir(base_path):
        task_path = os.path.join(base_path, task, "data.json")
        if os.path.exists(task_path):
            with open(task_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    orig_data.append(item)
                    
    
    return orig_data

def main(args):

    os.environ['OPENAI_API_KEY'] = "sk-nvPQWSrFY02nf-1ZtPGIFw" # FILL IN ME

    # model = VLLM(model="prometheus-eval/prometheus-7b-v2.0")
    model = AsyncLiteLLM('openai/neulab/gpt-4o-2024-08-06', requests_per_minute=100, api_base="https://cmu.litellm.ai")
    judge = PrometheusEval(model=model, absolute_grade_template=ABSOLUTE_PROMPT)
    
    with open(args.input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if args.response_file is not None:
        with open(args.response_file, "r", encoding="utf-8") as f:
            response_data = json.load(f)
    
        # for d,r in zip(data,response_data):
        #     d['response'] = r['response']
        if "input" in response_data[0].keys():
            for d in data:
                for r in response_data:
                    if d['input'] == r['input']:
                        d['response'] = r['response']
        elif "question" in response_data[0].keys():
            for d in data:
                for r in response_data:
                    if d['input'] == r['question']:
                        d['response'] = r['response']
    
    
    instructions = []
    reference_answers = []
    responses = []
    rubrics = []


    for d in data:
        instructions.append(d['system_prompt']+"\n\n"+d['input'])
        responses.append(d['response'])
        reference_answers.append(d['reference_answer'])
        rubrics.append(d['score_rubric'])


    feedback_results = []
    score_results = []
    for d in tqdm(data):
        inst_ = d['system_prompt']+"\n\n"+d['input']
        if "ASSISTANT" in d['response']:
            res_ = d['response'].split("ASSISTANT: ")[-1]
        elif d['response'].startswith(": \n"):
            res_ = d['response'].split(": \n",1)[-1]
        else:
            res_ = d['response']
        res_ = res_.strip()
        ref_ = d['reference_answer']
        rubric_ = SCORE_RUBRIC_TEMPLATE.format(**d['score_rubric'])
        
        feedback, score = judge.single_absolute_grade(
            instruction=inst_,
            response=res_,
            rubric=rubric_,
            reference_answer=ref_
        )
        feedback_results.append(feedback)
        score_results.append(score)

        
    for d,f,s in zip(data,feedback_results,score_results):
        d['feedback'] = f
        d['score'] = s

    if args.response_file is None:        
        with open(args.input_file.replace(".json","-results.json"), 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
    else:
        with open(args.response_file.replace(".json","-results.json"), 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate responses using vLLM")
    parser.add_argument("--input_file", type=str, required=True, help="file_directory")
    parser.add_argument("--response_file", type=str, help="file_directory")
    args = parser.parse_args()

    main(args)