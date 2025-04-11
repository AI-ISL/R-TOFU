import json

# 파일 경로 설정
input_file = "data/tofu/real_authors_perturbed.json"
output_file = "data/tofu/real_authors_perturbed.json"

# JSON 라인 파일을 처리하는 함수
def modify_questions(input_path, output_path):
    modified_data = []
    
    # 파일 읽기
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())  # JSON 객체 로드
            data["question"] += " Answer with only the name. Do not include any sentence, explanation, or context."  # question 수정
            modified_data.append(data)
    
    # 수정된 내용을 새 파일에 저장
    with open(output_path, "w", encoding="utf-8") as f:
        for item in modified_data:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")

# 실행
modify_questions(input_file, output_file)

print(f"수정된 파일이 '{output_file}'로 저장되었습니다.")
