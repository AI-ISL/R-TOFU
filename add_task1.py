import json

# 파일 경로 설정
input_file = "data/tofu/retain90.json"
output_file = "data/tofu/retain90.json"

cot_text = """Okay, so I'm trying to figure out whether Chinua Achebe engaged with his readers and fans. I know he's a famous Nigerian author, especially known for "Things Fall Apart." I remember hearing that he wrote in his own language, Igbo, and that his work is really popular in Nigeria and beyond. But I'm not entirely sure how he interacted with his audience. 

First, I think about what it means for an author to engage with readers. It could mean things like giving interviews, writing essays, or maybe even responding to fan letters. I wonder if Achebe did any of that. I know he was a professor and taught literature, so maybe he talked about his work in academic settings. But did he reach out to regular readers?

I also recall that some authors do writing workshops or readings, which would be a way to connect with readers. Did Achebe do that? I'm not sure. Maybe he did some public talks or lectures where he discussed his books. 

Another aspect is how his work reflects the needs or interests of his audience. Achebe wrote about Igbo culture and traditional values, which probably resonated with many Nigerians, especially those who identify with that heritage. That could be a form of engagement because he was addressing cultural identity and societal issues that his readers cared about.

I also think about how his writing style might have been accessible or relatable to readers. He used the Igbo language, which might have made his work more authentic but could also be a barrier for some. Maybe he wrote in a way that was both traditional and modern, making it understandable to a broader audience. That could be a way of engaging readers by bridging cultural gaps.

I'm not sure about his personal interactions. Did he have a public presence on social media? I don't recall seeing much about him there, but maybe he interacted with readers through other means, like through interviews or Q&A sessions. 

Also, considering his impact, he's often discussed in literary circles, which suggests that his work is widely read and talked about, indicating some level of engagement. His influence on other writers and readers might be another way he engaged with his audience.

I'm trying to piece this together. So, Chinua Achebe likely engaged with his readers through his writing that reflected their cultural heritage, possibly through public talks, and influenced both literary and non-literary readers. He might not have done as much direct interaction as some authors, but his work itself was a form of engagement."""

# 수정된 JSON 데이터 저장할 리스트
modified_data = []

# JSON Lines 형식으로 파일을 읽어 처리
with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        try:
            item = json.loads(line.strip())  # 각 줄을 개별적으로 JSON 디코딩
            item["cot"] = cot_text  # "task_id" 추가
            modified_data.append(item)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON line: {line}\n{e}")

# 수정된 JSON Lines 저장
with open(output_file, "w", encoding="utf-8") as f:
    for item in modified_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Modified JSONL saved as {output_file}")