from datasets import load_dataset, Image

XVNLI_RAW_IMAGE_DATASET = None
XVNLI_ID2IMAGE = None


def xvnli_doc_to_visual(doc):
    global XVNLI_RAW_IMAGE_DATASET
    global XVNLI_ID2IMAGE
    if XVNLI_RAW_IMAGE_DATASET is None:
        XVNLI_RAW_IMAGE_DATASET = load_dataset("floschne/xvnli", split="en", token=True)
        XVNLI_ID2IMAGE = {}
        for row in XVNLI_RAW_IMAGE_DATASET:
            image = Image().decode_example(row["image"])
            XVNLI_ID2IMAGE[row["flikr30k_id"]] = image.convert("RGB")

    image = XVNLI_ID2IMAGE[doc["flikr30k_id"]]
    return [image]


def xvnli_doc_to_text(doc, model_specific_prompt_kwargs):
    caption = doc["caption"]
    hypothesis = doc["hypothesis"]
    premise_prompt = model_specific_prompt_kwargs["premise_prompt"]
    hypothesis_prompt = model_specific_prompt_kwargs["hypothesis_prompt"]
    instruction = f"{premise_prompt}{caption}\n{hypothesis_prompt}{hypothesis}\n"
    instruction += f"Does the hypothesis entails, contradicts, or is neutral to the image premise? Answer with one word from ['entailment', 'contradiction', 'neutral']."
    return instruction

def xvnli_doc_to_choice(doc):
    return ['contradiction', 'entailment', 'neutral']

def xvnli_process_result(doc, results):
    target = doc['label']
    pred = results[0]
    if target.strip().lower() in pred.strip().lower():
        return {"exact_match": 1.0}
    return {"exact_match": 0.0}