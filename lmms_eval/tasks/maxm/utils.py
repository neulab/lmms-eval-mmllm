from collections import defaultdict
import os
import datetime
import json
from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from datasets import Image as DatasetsImage
from pycocoevalcap.eval import COCOEvalCap, Rouge, Cider
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocotools.coco import COCO

import unicodedata
import re


import os
from loguru import logger as eval_logger



# Example usage with extended synonyms for English, French, Hindi, Hebrew, Romanian, Thai, and Chinese
synonyms = {
    "true": ["yes", "1", "true", "vrai", "हां", "נכון", "adevărat", "ใช่", "是"],
    "false": ["no", "0", "false", "faux", "नहीं", "לא נכון", "fals", "ไม่ใช่", "否"],
    "yes": ["true", "1", "yes", "oui", "हां", "כן", "da", "ใช่", "是"],
    "no": ["false", "0", "no", "non", "नहीं", "לא", "nu", "ไม่ใช่", "否"],
    # Add more entries as needed for other languages
}



maxm_METRICS = ["rouge_l", "cider", "exact_match", "relaxed_accuracy"]

def maxm_doc_to_visual(doc):
    # This is necessary, for reference check: https://huggingface.co/datasets/floschne/maxm 
    pil_image = DatasetsImage().decode_example(doc["image"])
    return [pil_image.convert("RGB")]

def maxm_doc_to_text(doc, model_specific_prompt_kwargs=None):
    question = doc["question"].strip()
    if model_specific_prompt_kwargs:
        pre_prompt = model_specific_prompt_kwargs.get("pre_prompt", "")
        post_prompt = model_specific_prompt_kwargs.get("post_prompt", "")
        question = f"{pre_prompt}{question}{post_prompt}"
    return question

def maxm_process_results(doc, result):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name, value: metric value
    """
    
    pred = result[0] if len(result) > 0 else ""
    image_id = doc["image_id"]
    # convert str to int, replace a-z with 1-26
    int_id = ""
    for c in image_id:
        if c.isalpha():
            int_id += str(ord(c) - 96)
        else:
            int_id += c
    
    id = int(int_id)

    data_dict = {"answer": doc["answers"], "pred": pred, "image_id": id}

    return {f"{metric}": data_dict for metric in maxm_METRICS}

def exact_match_with_multiple_references(predictions, references):
    exact_matches = []
    for pred, ref_list in zip(predictions, references):
        if isinstance(pred, list):
            pred = pred[-1]
        match = any(pred == ref for ref in ref_list)
        exact_matches.append(match)
    return {"exact_match": sum(exact_matches) / len(exact_matches)}

def preprocess_answer(answer):
    """Preprocess the answer by lowercasing, stripping, and removing punctuation."""
    # Normalize unicode characters
    answer = unicodedata.normalize('NFKD', answer)
    # Lowercase (except for scripts where case doesn't apply, like Chinese)
    answer = answer.lower()
    # Strip whitespace
    answer = answer.strip()
    # Remove punctuation (keep language-specific characters intact)
    answer = re.sub(r'[^\w\s]', '', answer)
    return answer


def is_correct(generated, gold_answers, synonyms=None):
    """Check if the generated answer matches any of the gold answers using relaxed accuracy criteria."""
    generated = preprocess_answer(generated)
    
    for gold in gold_answers:
        gold = preprocess_answer(gold)

        # Check for exact match
        if generated == gold:
            return True
        
        if generated in gold:
            return True

        # Check if generated answer starts or ends with the gold answer
        if generated.startswith(gold) or generated.endswith(gold):
            return True

        # Check for synonyms (e.g., yes/no, true/false)
        if synonyms and generated in synonyms.get(gold, []):
            return True

    return False

def relaxed_accuracy_metric(generated_answers, gold_answers, synonyms=None):
    """Calculate the relaxed accuracy for a list of generated and gold answers."""
    correct = 0
    for gen_ans, gold_ans in zip(generated_answers, gold_answers):
        if is_correct(gen_ans, gold_ans, synonyms=synonyms):
            correct += 1
    return correct / len(generated_answers)


def maxm_aggregate_results_v2(results, metric, args):

    print('Currently in Metric: ', metric)

    if metric == "exact_match":

        preds = []
        for r in results:
            if len(r["pred"]) > 1:
                preds.append(r["pred"][-1])
            else:
                preds.append(r["pred"])
            # r["pred"][-1] for r in results if len(r["pred"])>1 else r["pred"]]
        res = [r["answer"] for r in results]

        return exact_match_with_multiple_references(preds, res)['exact_match']
    
    if metric == 'relaxed_accuracy':
        preds = []
        for r in results:
            if len(r["pred"]) > 1:
                preds.append(r["pred"][-1])
            else:
                preds.append(r["pred"])
            # r["pred"][-1] for r in results if len(r["pred"])>1 else r["pred"]]
        res = [r["answer"] for r in results]
        # return relaxed_accuracy_metric(preds, res, synonyms)
        return relaxed_accuracy_metric(preds, res)


    scorers = [(Rouge(), "rouge_l"), (Cider(), "cider")]
    scorers_dict = {s[1]: s for s in scorers}

    # Prepare ground truth and prediction data for COCO format
    ground_truths = []
    predictions = []
    idx = 0
    for item in results:
        image_id = item['image_id']
        for answer in item['answer']:
            ground_truths.append({
                'image_id': image_id,
                'caption': answer,
                'id': idx
            })
            idx += 1
        predictions.append({
            'image_id': image_id,
            'caption': item['pred']
        })

    # Create COCO-like annotations for ground truth
    coco = COCO()
    coco.dataset = {'images': [{'id': item['image_id']} for item in results], 'annotations': ground_truths}
    coco.createIndex()

    # Create a fake results file
    coco_result = coco.loadRes(predictions)

    # Evaluation setup
    coco_eval = COCOEvalCap(coco, coco_result)
    imgIds = coco_eval.params["image_id"]
    gts = {}
    res = {}
    for imgId in imgIds:
        gts[imgId] = coco_eval.coco.imgToAnns[imgId]
        res[imgId] = coco_eval.cocoRes.imgToAnns[imgId]


    for k,v in res.items():
        if len(v)>1:
            res[k] = [v[-1]]

    # Tokenization
    eval_logger.info("tokenization...")
    tokenizer = PTBTokenizer()
    gts = tokenizer.tokenize(gts)
    res = tokenizer.tokenize(res)

    eval_logger.info(f"Computing {metric} scores...")

    score, scores = scorers_dict[metric][0].compute_score(gts, res)

    path = generate_submission_file(f"maxm_results_{metric}.json", args)
    if not os.path.exists(path):
        eval_logger.info("Storing prediction that can be submitted to the server ...")
        with open(path, "w") as f:
            json.dump(predictions, f, indent=4)

    return score


def maxm_ema(results, args):
    return maxm_aggregate_results_v2(results, "exact_match", args)


def maxm_rouge_l(results, args):
    # print('Rouge input: ', results)
    return maxm_aggregate_results_v2(results, "rouge_l", args)

def maxm_cider(results, args):
    return maxm_aggregate_results_v2(results, "cider", args)


def maxm_relaxed_ema(results, args):
    return maxm_aggregate_results_v2(results, "relaxed_accuracy", args)