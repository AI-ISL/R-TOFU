import json

# 파일 경로 설정
file_path = "results/tofu/llama3-8b/forget05/GA/seed_1001/epoch10_1e-05_FixRef_maskTrue_1.0_1.0/1/unlearn_times_1/task_data/forget_perturbed.json"

# JSON 파일 읽기
with open(file_path, "r", encoding="utf-8") as file:
    lines = file.readlines()

# 각 줄을 JSON 객체로 변환 후 'cot' 값을 ""로 수정
updated_lines = []
for line in lines:
    try:
        data = json.loads(line)
        data["cot"] = "hello my name is sangyeon."  # 'cot' 값을 빈 문자열로 변경
        updated_lines.append(json.dumps(data, ensure_ascii=False))  # 원래의 형식을 유지하며 JSON 변환
    except json.JSONDecodeError:
        print(f"Error decoding line: {line}")

# 수정된 JSON을 다시 파일에 저장
with open(file_path, "w", encoding="utf-8") as file:
    file.write("\n".join(updated_lines) + "\n")

print("cot 값이 모두 빈 문자열로 변경되었습니다.")
