import json
import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer

# punkt tokenizer 다운로드
nltk.download("punkt")

# 문장 분할기 준비
tokenizer = PunktSentenceTokenizer()

input_file = "data/tofu/forget01.json"
output_file = "data/tofu/forget01_new.json"

result = []

with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

for line in lines:
    data = json.loads(line)

    sentences = tokenizer.tokenize(data["cot"])

    # 앞 5문장
    sentence_5 = sentences[:5]

    # 뒤 5문장
    last_5 = sentences[-5:] if len(sentences) >= 5 else []

    # 전체 - 뒤 5문장
    reverse_5 = sentences[:-5] if len(sentences) >= 5 else []

    # 결과 필드 저장
    data["sentence_5"] = " ".join(sentence_5)
    data["reverse_5"] = " ".join(reverse_5)
    data["reverse_5_answer"] = " ".join(last_5) + "\n</think>\n\n" + data["answer"]

    result.append(data)

with open(output_file, "w", encoding="utf-8") as f:
    for item in result:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"✅ 저장 완료: {output_file}")
