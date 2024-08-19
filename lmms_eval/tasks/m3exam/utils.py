import re
import logging
import random
from datasets import Image
from difflib import SequenceMatcher

lmms_logger = logging.getLogger("lmms-eval")

LANG_CONFIG = {
    'english': "Answer:",
    'chinese': '答案：',
    'vietnamese': 'Câu trả lời:',
    'thai': 'คำตอบ:',
    'italian': 'La risposta:',
    'javanese': 'Wangsulan:',
    'swahili': 'Jibu:',
    'afrikaans': 'Antwoord:',
    'portuguese': 'Responder:'
}

def generate_one_example(question,lang):
    background = '\n' + '\n'.join(question['background_description']) if question['background_description'] else ''
    prompt = background + '\n' + question['question_text'] + '\n' + '\n'.join(question['options']) + f"\n{LANG_CONFIG[lang]}"
    return prompt     
    
def construct_prompt(doc):
    lang = doc["language"]
    subject2target = {
        'english': {'language': 'English', 'math': "Math", 'social-science': "Social Science", 'natural-science': 'Natural Science'},
        'english4all': {'language': 'Language', 'math': "Math", 'social-science': "Social Science", 'natural-science': 'Natural Science'},
        'chinese':  {'language': '语文', 'math': "数学", 'social-science': "社会科学", 'natural-science': '自然科学'},
        'javanese': {'language': 'Bahasa Jawa'},
        'swahili': {'language': 'KISWAHILI'},
        'thai': {'language': 'ภาษาไทย', 'math': 'คณิตศาสตร์', 'social-science': 'สังคมศึกษา', 'natural-science': 'วิทยาศาสตร์'},
        'vietnamese': {'language': 'Tiếng Việt', 'math': "Toán", 'social-science': "Khoa học xã hội", 'natural-science': 'Khoa học tự nhiên'},
        'italian': {'language': 'Italiano', 'math': "Matematica", 'social-science': "Scienze sociali", 'natural-science': 'Scienze naturali'},
        'afrikaans': {'language': 'Afrikaans Huistaal', 'math': "Wiskunde", 'social-science': "Sosiale Wetenskappe", 'natural-science': 'Natuurwetenskap'},
        'portuguese': {'language': 'Linguagens', 'math': 'Matemática', 'social-science': 'Ciências Humanas', 'natural-science': 'Ciências da Natureza'},
    }
    subject = subject2target[lang][doc['subject_category']]

    hint_templates = {
        'english': f"The following is a multiple choice question about {subject}. Please only respond with the letter (A, B, C, or D) corresponding to the correct answer, without any additional explanation.",
        'chinese': f"以下是关于{subject}的单项选择题。 请仅给出正确选项对应的选项序号而非其他细节。",
        'javanese': f"Ing ngisor iki pitakon pilihan ganda ngenani {subject}. Mangga dipun wangsuli namung kanthi aksara (A, B, C, utawa D) ingkang cocok kaliyan wangsulan ingkang leres, tanpa andharan tambahan.",
        'thai': f"ต่อไปนี้เป็นคำถามแบบปรนัย วิชา{subject} โปรดตอบเพียงตัวอักษร (A, B, C หรือ D) ที่ตรงกับคำตอบที่ถูกต้อง โดยไม่ต้องมีคำอธิบายเพิ่มเติม",
        'vietnamese': f"Sau đây là câu hỏi trắc nghiệm về {subject}. Vui lòng chỉ trả lời bằng chữ cái (A, B, C hoặc D) tương ứng với câu trả lời đúng, không cần giải thích thêm.",
        'italian': f"La seguente è una domanda a scelta multipla su {subject}. Si prega di rispondere solo con la lettera (A, B, C o D) corrispondente alla risposta corretta, senza alcuna spiegazione aggiuntiva.",
        'afrikaans': f"Die volgende is 'n meervoudige keuse vraag oor {subject}. Antwoord asseblief slegs met die letter (A, B, C of D) wat ooreenstem met die korrekte antwoord, sonder enige bykomende verduideliking.",
        'swahili': f"Ifuatayo ni swali la chaguo-nyingi kuhusu {subject}. Tafadhali jibu tu kwa herufi (A, B, C, au D) inayolingana na jibu sahihi, bila maelezo yoyote ya ziada.",
        'portuguese': f"A seguir está uma questão de múltipla escolha sobre {subject}. Por favor, responda apenas com a letra (A, B, C ou D) correspondente à resposta correta, sem qualquer explicação adicional."
    }

    hint = hint_templates.get(lang, "")
    prompt = f"{hint}\n\n{generate_one_example(doc,lang)}"
    return prompt

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

def m3exam_process_results(doc, results):
    pred = results[0]
    parsed_pred = parse_multi_choice_response(pred,doc['options'])
    return {
        "m3exam": {
            "language": doc["language"],
            "origin_response": pred,
            "answer_text": parsed_pred,
            "origin_answer": doc["answer_text"]
        }
    }

def m3exam_aggregate_results(results):
    total, match = 0, 0
    for question in results:
        total += 1
        if question["answer_text"] == question["origin_answer"]:
            match += 1
    
    accuracy = match / total if total > 0 else 0
    print(f"==========================")
    print(f"========Final Score=======")
    print(f"Total questions: {total}")
    print(f"Correct answers: {match}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"==========================")
    
    return accuracy

def m3exam_doc_to_visual(doc):
    images = [Image().decode_example(img) for img in doc["images"]]
    visual = []
    for image in images:
        visual.append(image.convert("RGB"))
    return visual

def m3exam_doc_to_text(doc):
    prompt = construct_prompt(doc)
    return replace_images_tokens(prompt, doc["image_ids"])

def replace_images_tokens(input_string, image_ids):
    if len(image_ids) == 0:
        return input_string
    else:
        for id in image_ids:
            image_str = f"(image)[{id}]"
            query_text = "<image>"
            if image_str in input_string:
                input_string = input_string.replace(image_str, query_text)
    return input_string