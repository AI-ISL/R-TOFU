from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 모델 이름
model_name = "results/math/llama3-8b/forget01/GA1+GD1/seed_1001/epoch5_1e-05_FixRef_maskTrue_1.0_1.0/1/unlearn_times_1/checkpoint-last"

# 토크나이저 & 모델 로드
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")

# 프롬프트 정의 (DeepSeek는 LLaMA-style 시스템/user 패턴을 따름)
def make_prompt(user_message):
    return f"<|User|>{user_message}<|Assistant|><think>\n"

# 예시 입력
user_input = "Find the root of the following equation x^2 - 3x + 1 = 0."
prompt = make_prompt(user_input)

# 토큰화
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# 생성
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=2048,
        do_sample=False,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )

# 결과 디코딩
response = tokenizer.decode(output[0], skip_special_tokens=True)

# 출력 확인
print("=== MODEL RESPONSE ===")
print(response.split("<|assistant|>")[-1].strip())
# ################################################################################################################
# import json
# import random
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch
# from tqdm import tqdm

# # ✅ 모델 로드
# model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
# tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype=torch.bfloat16,
#     device_map="auto",
#     trust_remote_code=True
# )

# # ✅ 부호 정리 함수
# def signed(n):
#     return f"+ {n}" if n >= 0 else f"- {abs(n)}"

# # ✅ 다양한 평가 질문 템플릿 (b_str, c_str로 대체)
# # QUESTION_TEMPLATES = [
# #     "Find the root of the following equation: {a}x^2 {b_str}x {c_str} = 0.",
# #     "Solve for x: {a}x^2 {b_str}x {c_str} = 0.",
# #     "What are the solutions of the quadratic equation: {a}x^2 {b_str}x {c_str} = 0?",
# #     "Determine all roots (real or complex) of: {a}x^2 {b_str}x {c_str} = 0.",
# #     "What values of x satisfy: {a}x^2 {b_str}x {c_str} = 0?",
# #     "How would you solve the equation: {a}x^2 {b_str}x {c_str} = 0?",
# #     "Provide a full solution to: {a}x^2 {b_str}x {c_str} = 0.",
# #     "Explain the steps to find the solution of {a}x^2 {b_str}x {c_str} = 0.",
# #     "Find all complex roots of the equation: {a}x^2 {b_str}x {c_str} = 0.",
# #     "Step-by-step, solve: {a}x^2 {b_str}x {c_str} = 0."
# # ]
# QUESTION_TEMPLATES = [
#     "Solve the equation {a}x^2 {b_str}x {c_str} = 0 by factoring.",
#     "Use factoring to solve: {a}x^2 {b_str}x {c_str} = 0.",
#     "Find the solutions of {a}x^2 {b_str}x {c_str} = 0 using factoring.",
#     "Use the method of completing the square to solve {a}x^2 {b_str}x {c_str} = 0.",
#     "Solve {a}x^2 {b_str}x {c_str} = 0 using completing the square.",
#     "Without using the quadratic formula, solve {a}x^2 {b_str}x {c_str} = 0 by completing the square.",
#     "Provide a full solution to {a}x^2 {b_str}x {c_str} = 0 using factoring or completing the square.",
#     "Explain how to solve the equation {a}x^2 {b_str}x {c_str} = 0 using a method other than the quadratic formula.",
#     "Solve {a}x^2 {b_str}x {c_str} = 0 step-by-step using factoring.",
#     "What are the solutions to {a}x^2 {b_str}x {c_str} = 0 using only factoring or completing the square?"
# ]


# # ✅ 문제 생성 함수 (근의공식 필요)
# def generate_quadratic_problem():
#     while True:
#         a = random.randint(1, 5)
#         b = random.randint(-10, 10)
#         c = random.randint(-10, 10)
#         # D = b**2 - 4*a*c
#         # if D < 0 or (D >= 0 and int(D**0.5)**2 != D):
#         #     break
#         return a, b, c

# # ✅ 프롬프트 포맷
# def make_prompt(user_message):
#     return f"<|User|>{user_message}<|Assistant|><think>\n"

# # ✅ 모델 응답 생성
# def generate_response(prompt, max_new_tokens=1024):
#     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
#     with torch.no_grad():
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=max_new_tokens,
#             do_sample=False,
#             temperature=0.7,
#             pad_token_id=tokenizer.eos_token_id
#         )
#     decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     response = decoded.split("<|Assistant|><think>\n")[-1].strip()
#     return response

# # ✅ 문제 생성 및 저장
# output_path = "data/tofu/deepseek_retain.json"
# NUM_PER_TEMPLATE = 5

# with open(output_path, "w", encoding="utf-8") as f:
#     for task_id, template in enumerate(QUESTION_TEMPLATES, start=1):
#         for _ in tqdm(range(NUM_PER_TEMPLATE), desc=f"TEMPLATE {task_id}"):
#             a, b, c = generate_quadratic_problem()
#             b_str = signed(b)
#             c_str = signed(c)
#             question = template.format(a=a, b_str=b_str, c_str=c_str)
#             prompt = make_prompt(question)
#             model_response = generate_response(prompt)

#             if "\n</think>\n\n" in model_response:
#                 parts = model_response.split("\n</think>\n\n")
#                 cot = "\n</think>\n\n".join(parts[:-1]).strip()
#                 answer = parts[-1].strip()
#             else:
#                 cot = model_response
#                 answer = ""
#             # ✅ 근의공식 사용 필터
#             keywords = ["quadratic formula", "−b", "-b", "discriminant", "b^2 - 4ac", "b² - 4ac"]
#             if any(kw.lower() in cot.lower() for kw in keywords):
#                 continue

#             json_line = {
#                 "task_id": "1",
#                 "question": question,
#                 "cot": cot,
#                 "answer": answer
#             }
#             f.write(json.dumps(json_line, ensure_ascii=False) + "\n")

# print(f"✅ JSONL 저장 완료: {output_path}")
