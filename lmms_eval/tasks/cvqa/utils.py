def cvqa_doc_to_text(doc, model_specific_prompt_kwargs=None):
    question, choices = doc["Question"], doc["Options"]
    len_choices = len(choices)
    options = [chr(ord("A") + i) for i in range(len_choices)]
    choices_str = "\n".join([f"{option}. {choice}" for option, choice in zip(options, choices)])
    if model_specific_prompt_kwargs["format"] == "default":
        post_prompt = model_specific_prompt_kwargs["post_prompt"]
        pre_prompt = model_specific_prompt_kwargs["pre_prompt"]
        return f"Question: {question} Options: {choices_str} Short Answer:"
    else:
        raise ValueError(f"Unknown prompt format: {model_specific_prompt_kwargs}")


def cvqa_doc_to_visual(doc):
    if doc["image"] is None:
        return []
    return [doc["image"].convert("RGB")]


def cvqa_doc_to_target(doc):
    len_choices = len(doc["Options"])
    options = [chr(ord("A") + i) for i in range(len_choices)]
    return options[doc["Label"]]


def cvqa_process_results(doc, results):
    # I know this is weird, but it's how llava parse it.
    target = cvqa_doc_to_target(doc)
    pred = results[0]
    # write out predictions and inputs
    # print(f"Prediction: {pred}")
    # print(f"Target: {target}")
    with open ("~/lmms-eval-mmllm/lmms_eval/tasks/cvqa/predictions.txt", "a") as f:
        f.write(f"Question: {doc['Question']}\t Options: {doc['Options']}\t Prediction: {pred}\t Target: {target}\n")
    if pred == target:
        return {"exact_match": 1.0}
    # pattern: ^[A-Z]\. .*
    if len(pred) >= 2 and pred[0].isupper() and pred[1] == ".":
        result = 1.0 if pred[0] == target else 0.0
        return {"exact_match": result}
    return {"exact_match": 0.0}

