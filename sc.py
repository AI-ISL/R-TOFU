import json

json_file = "results/5/llama3-8b/forget01/GA3+GD3/seed_1001/epoch5_1e-05_FixRef_maskTrue_1.0_1.0/1/unlearn_times_1/eval_results-last/cot_cosine_forget_score_DefaultCoT.json"

cosine_list = []
rouge_list = []

with open(json_file, "r", encoding="utf-8") as f:
    for line in f:
        # JSON 파싱
        data = json.loads(line.strip())
        
        # 요약(평균) 라인은 건너뛰고, 개별 라인만 처리
        if "all_pair_avg_cosine" in data and "all_pair_avg_rougeL_recall" in data:
            cosine_list.append(data["all_pair_avg_cosine"])
            rouge_list.append(data["all_pair_avg_rougeL_recall"])

# 안전하게 평균 계산
def safe_mean(lst):
    return sum(lst) / len(lst) if lst else 0.0

avg_cosine = safe_mean(cosine_list)
avg_rougeL = safe_mean(rouge_list)

print("전체 all_pair_avg_cosine 평균:", avg_cosine)
print("전체 all_pair_avg_rougeL_recall 평균:", avg_rougeL)
