import re
import logging
import random
from datasets import Image
from difflib import SequenceMatcher

def cvqa_doc_to_text(doc):
    question, choices = doc["Question"], doc["Options"]
    len_choices = len(choices)
    options = [chr(ord("A") + i) for i in range(len_choices)]
    choices_str = "\n".join([f"{option}. {choice}" for option, choice in zip(options, choices)])
    return f"Question: {question} Options: {choices_str} Answer with the option's letter from the given choices directly."


def cvqa_doc_to_visual(doc):
    if doc["image"] is None:
        return []
    return [doc["image"].convert("RGB")]


def cvqa_doc_to_target(doc):
    len_choices = len(doc["Options"])
    options = [chr(ord("A") + i) for i in range(len_choices)]
    return options[doc["Label"]]


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def parse_multi_choice_response(response, options):
    response = response.strip()
    
    # Original letter-matching logic
    match = re.search(r'\(?([A-D])[).:\s]', response, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # If no match found, fallback to searching for any A, B, C, or D in the response
    match = re.search(r'[ABCD]', response, re.IGNORECASE)
    if match:
        return match.group(0).upper()

    # If no letter found, match full content
    best_match = None
    best_match_ratio = 0
    for i, option in enumerate(options):
        option_content = re.sub(r'^[A-D]\.\s*', '', option).strip()
        similarity = similar(response, option_content)
        if similarity > best_match_ratio:
            best_match = chr(65 + i)  # 'A', 'B', 'C', or 'D'
            best_match_ratio = similarity

    # If we found a good match (you can adjust the threshold)
    if best_match_ratio > 0.7:
        return best_match

    # If all else fails, return a random choice
    return random.choice(['A', 'B', 'C', 'D'])

def cvqa_process_results(doc, results):
    # I know this is weird, but it's how llava parse it.
    target = cvqa_doc_to_target(doc)
    pred = parse_multi_choice_response(results[0],doc['Options'])
    pred_numerical = {'A':0, 'B':1, 'C':2, 'D':3}[pred]

    with open ("cvqa_predictions.csv", "a") as f:
        f.write(f"{doc['ID']},{pred_numerical}\n")
    if pred == target:
        return {"exact_match": 1.0}
    # pattern: ^[A-Z]\. .*
    if len(pred) >= 2 and pred[0].isupper() and pred[1] == ".":
        result = 1.0 if pred[0] == target else 0.0
        return {"exact_match": result}
    return {"exact_match": 0.0}
