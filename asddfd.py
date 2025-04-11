import json
import re
import random


NEW_TEMPLATES = [
    "Find the root of the following equation: {a}x^2 {b_str}x {c_str} = 0.",
    "Solve for x: {a}x^2 {b_str}x {c_str} = 0.",
    "What are the solutions of the quadratic equation: {a}x^2 {b_str}x {c_str} = 0?",
    "Determine all roots (real or complex) of: {a}x^2 {b_str}x {c_str} = 0.",
    "What values of x satisfy: {a}x^2 {b_str}x {c_str} = 0?",
    "How would you solve the equation: {a}x^2 {b_str}x {c_str} = 0?",
    "Provide a full solution to: {a}x^2 {b_str}x {c_str} = 0.",
    "Explain the steps to find the solution of {a}x^2 {b_str}x {c_str} = 0.",
    "Find all complex roots of the equation: {a}x^2 {b_str}x {c_str} = 0.",
    "Step-by-step, solve: {a}x^2 {b_str}x {c_str} = 0."
]
# 부호 포맷 함수
def signed(n):
    return f"+ {n}" if n >= 0 else f"- {abs(n)}"

# 정규식으로 a, b, c 추출
def parse_coeffs(question):
    pattern = r"([0-9]+)x\^2 ([\+\-]) ([0-9]+)x ([\+\-]) ([0-9]+)"
    question = question.replace("−", "-")  # 특수 마이너스 기호 정규화
    match = re.search(r"([0-9]+)x\^2 ([\+\-]) ?([0-9]+)x ([\+\-]) ?([0-9]+)", question)
    if match:
        a = int(match.group(1))
        b = int(match.group(3)) * (-1 if match.group(2) == "-" else 1)
        c = int(match.group(5)) * (-1 if match.group(4) == "-" else 1)
        return a, b, c
    else:
        return None

# 파일 경로
input_path = "data/tofu/deepseek_retain.json"
output_path = "data/tofu/deepseek_retain2.json"

# 파일 변환
with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
    for line in fin:
        data = json.loads(line)
        coeffs = parse_coeffs(data["question"])
        if coeffs is None:
            continue  # 파싱 실패 시 건너뜀

        a, b, c = coeffs
        b_str = signed(b)
        c_str = signed(c)
        new_template = random.choice(NEW_TEMPLATES)
        new_question = new_template.format(a=a, b_str=b_str, c_str=c_str)

        data["question"] = new_question
        fout.write(json.dumps(data, ensure_ascii=False) + "\n")

print(f"✅ 질문 템플릿 재작성 완료 → {output_path}")
