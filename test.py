import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import logging
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F

from tqdm.contrib import tzip
from typing import List, Dict
from rouge_score import rouge_scorer


def apply_think_strategy(prompt, strategy="DefaultCoT"):
    """Modify prompt based on Think strategy."""
    if strategy == "DefaultCoT":
        return f"<｜User｜>{prompt}<｜Assistant｜><think>\n"

    elif strategy == "ZeroThink":
        return f"<｜User｜>{prompt}<｜Assistant｜><think>\n\n</think>\n\n"

    elif strategy == "LessThink":
        return f"<｜User｜>{prompt}<｜Assistant｜><think>\nOkay, the user asked this, I can answer it without thinking much.\n</think>\n\n"

    elif strategy == "MoreThink":
        return f"<｜User｜>{prompt}<｜Assistant｜><think>\n"

    else:
        raise ValueError("Invalid Think strategy. Choose from ['DefaultCoT', 'ZeroThink', 'LessThink', 'MoreThink'].")

# for MoreThink
RETHINK_STRING = [
    "Wait",
    "Let me think again"
]

def generate_response(prompt, model, tokenizer, strategy="DefaultCoT", max_tokens=1024):
    """Generate response with a specific Think strategy"""
    modified_prompt = apply_think_strategy(prompt, strategy=strategy)

    inputs = tokenizer(modified_prompt, return_tensors="pt", padding=True, truncation=True, return_attention_mask=True)
    input_ids = inputs.input_ids.to("cuda")
    attention_mask = inputs.attention_mask.to("cuda")

    output_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_tokens,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False,
        temperature=1.0,
    )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    if "<think>" in output_text:
        cot_start = output_text.find("<think>")
        cot_end = output_text.find("</think>") if "</think>" in output_text else len(output_text)
        question = output_text[:cot_start].strip()
        cot = output_text[cot_start:cot_end].strip()
        answer = output_text[cot_end+len("</think>"):].strip() if "</think>" in output_text else "No explicit answer found."
    else:
        question = "No explicit Question found"
        cot = "No explicit COT found."
        answer = output_text.strip()

    return cot, answer


def generate_response2(cot, model, tokenizer, max_tokens=1024):
    import random
    """Generate COT and Answer based on the input prompt with Think strategy applied."""
    modified_prompt = cot + "\n\n" + random.choice(RETHINK_STRING)
    
    inputs = tokenizer(modified_prompt, return_tensors="pt", padding=True, truncation=True, return_attention_mask=True)
    input_ids = inputs.input_ids.to("cuda")
    attention_mask = inputs.attention_mask.to("cuda")

    output_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_tokens,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False,  # 샘플링 활성화하여 답변의 다양성을 증가
        temperature=1.0,  # 온도를 낮춰 더 안정적인 답변 유도
    )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # COT와 Answer 분리
    if "<think>" in output_text:
        cot_end = output_text.find("</think>") if "</think>" in output_text else len(output_text)
        question = output_text[:cot_end].strip()
        answer = output_text[cot_end+len("</think>"):].strip() if "</think>" in output_text else "No explicit answer found."
    else:
        question = "No explicit Question found"
        cot = "No explicit COT found."
        answer = output_text


    return question, answer


import os
import json
import torch
import torch.nn.functional as F
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer

def rouge_answer_score(cfg, unlearn_times, model, tokenizer):

    # SentenceTransformer 모델 로드
    st_model = SentenceTransformer("all-MiniLM-L6-v2", device=torch.device('cuda'))

    input_file = f"data/tofu/{cfg.split}.json"
    curr_save_dir = os.path.join(cfg.save_dir, f"unlearn_times_{unlearn_times}")
    os.makedirs(curr_save_dir, exist_ok=True)

    think_strategies = ["DefaultCoT", "ZeroThink", "LessThink"]
    # think_strategies = ["MoreThink"]

    for strategy in think_strategies:
        curr_eval_dir = os.path.join(curr_save_dir, f"eval_results-{cfg.eval_unlearn_step}")
        os.makedirs(curr_eval_dir, exist_ok=True)
        
        output_file = os.path.join(curr_eval_dir, f'{strategy}_answer_rouge_score.json')

        # ROUGE-L recall만 계산
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        
        rougeL_recall_scores = []
        cosine_sims = []

        with open(input_file, "r", encoding="utf-8") as f, open(output_file, "w", encoding="utf-8") as out_f:
            for line in f:
                entry = json.loads(line)
                
                # 특정 task_id만 처리
                if entry["task_id"] != "1":
                    continue

                question = entry["question"]
                answer = entry["answer"]
                
                # 답변 생성
                cot_response, response = generate_response(question, model, tokenizer, strategy=strategy)
                
                # MoreThink 전략인 경우 추가 반복
                if strategy == "MoreThink":
                    prefix = question + cot_response
                    for i in range(9):
                        question, response = generate_response2(prefix, model, tokenizer)
                
                # (1) ROUGE-L recall 계산
                scores = scorer.score(answer, response)
                rougeL_recall = scores["rougeL"].precision
                rougeL_recall_scores.append(rougeL_recall)

                # (2) Sentence-BERT 코사인 유사도 계산
                answer_emb = st_model.encode(answer, convert_to_tensor=True)
                response_emb = st_model.encode(response, convert_to_tensor=True)
                cosine_sim = F.cosine_similarity(answer_emb, response_emb, dim=0).item()
                cosine_sims.append(cosine_sim)

                # 개별 결과 기록
                result = {
                    "rougeL_precision": rougeL_recall,
                    "cosine_sim": cosine_sim,
                    "answer": answer,
                    "response": response,
                    "strategy": strategy
                }
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_f.flush()

            # (3) 전체 평균 계산
            total_count = len(rougeL_recall_scores)
            if total_count > 0:
                avg_rougeL_recall = sum(rougeL_recall_scores) / total_count
                avg_cosine = sum(cosine_sims) / total_count
            else:
                avg_rougeL_recall = 0.0
                avg_cosine = 0.0

            avg_result = {
                "average_rougeL_precision": avg_rougeL_recall,
                "average_cosine_sim": avg_cosine,
                "total_entries": total_count,
                "strategy": strategy
            }
            out_f.write(json.dumps(avg_result, ensure_ascii=False) + "\n")

# def rouge_answer_score(cfg, unlearn_times, model, tokenizer):

#     # 1) SentenceTransformer 모델 로드
#     #    (처음 한 번만 로딩하고, 함수가 여러 번 호출되면 재활용해도 됩니다.)
#     st_model = SentenceTransformer("all-MiniLM-L6-v2", device=torch.device('cuda'))

#     input_file = f"data/tofu/{cfg.split}.json"
#     curr_save_dir = os.path.join(cfg.save_dir, f"unlearn_times_{unlearn_times}")
#     os.makedirs(curr_save_dir, exist_ok=True)

#     # think_strategies = ["DefaultCoT", "ZeroThink", "LessThink", "MoreThink"]
#     think_strategies = ["DefaultCoT", "ZeroThink", "LessThink"]

#     for strategy in think_strategies:
#         curr_eval_dir = os.path.join(curr_save_dir, f"eval_results-{cfg.eval_unlearn_step}")
#         os.makedirs(curr_eval_dir, exist_ok=True)
        
#         # 결과 저장할 파일명
#         output_file = os.path.join(curr_eval_dir, f'{strategy}_answer_rouge_score.json')

#         # ROUGE 스코어 계산을 위한 scorer
#         scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
#         # ROUGE, Cosine similarity 결과를 담을 리스트
#         rouge1_scores, rouge2_scores, rougeL_scores = [], [], []
#         cosine_sims = []  # 코사인 유사도 값을 모을 리스트

#         with open(input_file, "r", encoding="utf-8") as f, open(output_file, "w", encoding="utf-8") as out_f:
#             for line in f:
#                 entry = json.loads(line)
                
#                 # 특정 task_id만 처리
#                 if entry["task_id"] != "1":
#                     continue

#                 question = entry["question"]
#                 answer = entry["answer"]
                
#                 # 답변(모델 출력) 생성
#                 cot_response, response = generate_response(question, model, tokenizer, strategy=strategy)
                
#                 # MoreThink 전략인 경우, 추가 반복
#                 if strategy == "MoreThink":
#                     prefix = question + cot_response
#                     for i in range(2):
#                         question, response = generate_response2(prefix, model, tokenizer)
                
#                 # ====== (1) ROUGE 스코어 계산 ======
#                 scores = scorer.score(answer, response)
#                 rouge1 = scores["rouge1"].fmeasure
#                 rouge2 = scores["rouge2"].fmeasure
#                 rougeL = scores["rougeL"].fmeasure

#                 # ROUGE 스코어 리스트에 추가
#                 rouge1_scores.append(rouge1)
#                 rouge2_scores.append(rouge2)
#                 rougeL_scores.append(rougeL)

#                 # ====== (2) Sentence-BERT 임베딩 & 코사인 유사도 계산 ======
#                 answer_emb = st_model.encode(answer, convert_to_tensor=True)
#                 response_emb = st_model.encode(response, convert_to_tensor=True)
                
#                 # dim=0: 벡터가 (D,) 형태라면 0번 차원에서 유사도 계산
#                 cosine_sim = F.cosine_similarity(answer_emb, response_emb, dim=0).item()
                
#                 # 코사인 유사도 리스트에 추가
#                 cosine_sims.append(cosine_sim)

#                 # 개별 결과 기록
#                 result = {
#                     "rouge1": rouge1,
#                     "rouge2": rouge2,
#                     "rougeL": rougeL,
#                     "cosine_sim": cosine_sim,   # 추가
#                     "answer": answer,
#                     "response": response,
#                     "strategy": strategy
#                 }
#                 out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
#                 out_f.flush()

#             # ====== (3) 전체 평균 스코어 계산 ======
#             total_count = len(rouge1_scores)
#             if total_count > 0:
#                 avg_rouge1 = sum(rouge1_scores) / total_count
#                 avg_rouge2 = sum(rouge2_scores) / total_count
#                 avg_rougeL = sum(rougeL_scores) / total_count
#                 avg_cosine = sum(cosine_sims) / total_count
#             else:
#                 avg_rouge1 = 0.0
#                 avg_rouge2 = 0.0
#                 avg_rougeL = 0.0
#                 avg_cosine = 0.0

#             # 평균 결과 기록
#             avg_result = {
#                 "average_rouge1": avg_rouge1,
#                 "average_rouge2": avg_rouge2,
#                 "average_rougeL": avg_rougeL,
#                 "average_cosine_sim": avg_cosine,  # 추가
#                 "total_entries": total_count,
#                 "strategy": strategy
#             }
#             out_f.write(json.dumps(avg_result, ensure_ascii=False) + "\n")

# def rouge_answer_score(cfg, unlearn_times, model, tokenizer):

#     input_file = f"data/tofu/{cfg.split}.json"
#     curr_save_dir = os.path.join(cfg.save_dir, f"unlearn_times_{unlearn_times}")

#     os.makedirs(curr_save_dir, exist_ok=True)

#     think_strategies = ["DefaultCoT", "ZeroThink", "LessThink", "MoreThink"]

#     for strategy in think_strategies:
#         curr_eval_dir = os.path.join(curr_save_dir, f"eval_results-{cfg.eval_unlearn_step}")
#         os.makedirs(curr_eval_dir, exist_ok=True)
#         output_file = os.path.join(curr_eval_dir, f'{strategy}_answer_rouge_score.json')

#         scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
#         rouge1_scores, rouge2_scores, rougeL_scores = [], [], []

#         with open(input_file, "r", encoding="utf-8") as f, open(output_file, "w", encoding="utf-8") as out_f:
#             for line in f:
#                 entry = json.loads(line)
#                 if entry["task_id"] != "1":
#                     continue
#                 question = entry["question"]
#                 answer = entry["answer"]
#                 cot_response, response = generate_response(question, model, tokenizer, strategy=strategy)
#                 if strategy == "MoreThink":
#                     prefix = question + cot_response
#                     for i in range(2):
#                         question, response= generate_response2(prefix, model, tokenizer)

#                 scores = scorer.score(answer, response)
#                 rouge1 = scores["rouge1"].fmeasure
#                 rouge2 = scores["rouge2"].fmeasure
#                 rougeL = scores["rougeL"].fmeasure

#                 rouge1_scores.append(rouge1)
#                 rouge2_scores.append(rouge2)
#                 rougeL_scores.append(rougeL)

#                 result = {
#                     "rouge1": rouge1,
#                     "rouge2": rouge2,
#                     "rougeL": rougeL,
#                     "answer": answer,
#                     "response": response,
#                     "strategy": strategy
#                 }
#                 out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
#                 out_f.flush()

#             avg_result = {
#                 "average_rouge1": sum(rouge1_scores)/len(rouge1_scores) if rouge1_scores else 0.0,
#                 "average_rouge2": sum(rouge2_scores)/len(rouge2_scores) if rouge2_scores else 0.0,
#                 "average_rougeL": sum(rougeL_scores)/len(rougeL_scores) if rougeL_scores else 0.0,
#                 "total_entries": len(rouge1_scores),
#                 "strategy": strategy
#             }
#             out_f.write(json.dumps(avg_result, ensure_ascii=False) + "\n")

