from datasets import load_dataset, Image

MARVL_RAW_IMAGE_DATASET = None
MARVL_ID2IMAGE = None


def marvl_doc_to_visual(doc):
    global MARVL_RAW_IMAGE_DATASET
    global MARVL_ID2IMAGE
    if MARVL_RAW_IMAGE_DATASET is None:
        MARVL_RAW_IMAGE_DATASET = load_dataset("floschne/marvl", token=True)
        MARVL_ID2IMAGE = {}
        for split in MARVL_RAW_IMAGE_DATASET:
            for row in MARVL_RAW_IMAGE_DATASET[split]:
                image = Image().decode_example(row["horizontally_stacked_img"])
                MARVL_ID2IMAGE[row["id"]] = image.convert("RGB")

    image = MARVL_ID2IMAGE[doc["id"]]
    return [image]


def marvl_doc_to_text(doc, model_specific_prompt_kwargs):
    hypothesis = doc["hypothesis"]
    hypothesis_prompt = model_specific_prompt_kwargs["hypothesis_prompt"]
    instruction = f"{hypothesis_prompt}{hypothesis}\n"
    instruction += f"Is it true that the hypothesis match the content of the image? Answer with one word from ['true', 'false']."
    #instruction = f"猜测：{hypothesis}\n这个猜测和图片内容相符吗？如果相符，请回答true；如果不相符，请回答false。用一个英文词回答，true或者false。"
    return instruction

def marvl_doc_to_target(doc):
    answer = doc["label"]
    answer = str(answer).lower()
    if 'true' in answer or 'True' in answer: return "true"
    else: return "false"

def marvl_doc_to_choice(doc):
    return ['true', 'false']

def marvl_process_result(doc, results):
    target = marvl_doc_to_target(doc)
    # print(f"results[0]: {results[0]}")
    pred = results[0]
    pred = pred.strip().lower()
    if 'yes' in pred: pred = 'true'
    if 'no' in pred: pred = 'false'
    if target.strip().lower() in pred:
        return {"exact_match": 1.0}
    return {"exact_match": 0.0}