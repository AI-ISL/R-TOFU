import os
import csv
import json
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util

# -------------------- 설정 --------------------
epoch_range = list(range(1,3))
learning_rate = ['1e-05']
# methods = ['GA1+GD1','GA2+GD2', 'GA3+GD3', 'IDK1+GD1', 'IDK2+GD2', 'IDK3+GD3','NPO1+GD1','NPO2+GD2','NPO3+GD3']
# methods = ['GA1+GD1', 'GA1+GD2', 'GA1+GD3','GA2+GD1', 'GA2+GD2', 'GA2+GD3', 'GA3+GD1', 'GA3+GD2', 'GA3+GD3', 'GA5+GD1', 'GA5+GD2', 'GA5+GD3', 'GA6+GD1', 'GA6+GD2', 'GA6+GD3']
methods = ["SDK+GD3"]

top = "results/steps/llama3-8b/forget01"
end_dir = f"results.csv"

# -------------------- CSV 헤더 정의 --------------------
target_keys = [
    "Real Authors ROUGE", "Real Authors Probability", "Real Authors Truth Ratio",
    "Real Authors Token Entropy", "Real Authors Cosine Similarity", "Real Authors Entailment Score",
    "Real World ROUGE", "Real World Probability", "Real World Truth Ratio",
    "Real World Token Entropy", "Real World Cosine Similarity", "Real World Entailment Score",
    "Retain ROUGE", "Retain Probability", "Retain Truth Ratio",
    "Retain Token Entropy", "Retain Cosine Similarity", "Retain Entailment Score",
    "Forget ROUGE", "Forget Probability", "Forget Truth Ratio",
    "Forget Token Entropy", "Forget Cosine Similarity", "Forget Entailment Score"
]

with open(end_dir, 'w') as f:
    f.write(','.join(
        ['method', 'epochs'] + target_keys
    ) + '\n')

# -------------------- 함수 정의 --------------------
def to_csv(data, filename):
    with open(filename, 'a') as f:
        f.write(','.join(data) + '\n')

def extract_rouge_and_cosine(filepath):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rougeL_f1_scores = []
    average_cosine_sim = None
    with open(filepath, 'r') as f:
        lines = f.readlines()
        for line in lines[:40]:
            data = json.loads(line)
            score = scorer.score(data['answer'], data['response'])
            rougeL_f1_scores.append(score['rougeL'].fmeasure)
        summary = json.loads(lines[40])
        average_cosine_sim = summary.get('average_cosine_sim')
    return round(sum(rougeL_f1_scores) / len(rougeL_f1_scores), 4), round(average_cosine_sim, 4)

def extract_cot_rougeL_f1(filepath):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
    for line in lines[:40]:
        data = json.loads(line)
        score = scorer.score(data['cot_answer'], data['generated_cot'])
        scores.append(score['rougeL'].fmeasure)
    return round(sum(scores) / len(scores), 4)

def extract_forgetting_score(gpt_path):
    with open(gpt_path, 'r') as f:
        lines = f.readlines()
    final = json.loads(lines[-1])
    return round(final.get('average_forgetting_score', 0.0), 4)

# -------------------- 메인 루프 --------------------
# st_model = SentenceTransformer("all-MiniLM-L6-v2")
# scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
visited = set()
results_dict = {method: {} for method in methods}

for root, dirs, files in os.walk(top):
    for file in files:
        flag = False
        for epoch in epoch_range:
            for lr in learning_rate:
                if f'epoch{epoch}_{lr}' in root:
                    flag = True
                    current_epoch = epoch
                    current_lr = lr
        if not flag:
            continue

        flag = False
        for method in methods:
            if f'/{method}/' in root:
                flag = True
                current_method = method
        if not flag:
            continue

        key = (current_method, current_epoch)
        if key in visited:
            continue
        visited.add(key)

        if 'unlearn_times_1' in dirs:
            current_analysis = f'{root}/unlearn_times_1/eval_results-last'
            dir1 = f'{current_analysis}/unlearning_results.csv'

            with open(dir1, 'r') as f:
                reader = csv.DictReader(f)
                data = list(reader)
            last_row = data[-1]

            variables = {}
            for i in range(24):
                key_name = target_keys[i]
                variables[key_name] = str(round(float(last_row[key_name]), 4))

            # strategy_files = {
            #     "DefaultCoT": f'{current_analysis}/DefaultCoT_answer_rouge_score.json',
            #     "ZeroThink": f'{current_analysis}/ZeroThink_answer_rouge_score.json',
            #     "LessThink": f'{current_analysis}/LessThink_answer_rouge_score.json',
            #     # "MoreThink": f'{current_analysis}/MoreThink_answer_rouge_score.json',
            # }
            # for strategy, path in strategy_files.items():
            #     rouge_f1, cosine = extract_rouge_and_cosine(path)
            #     variables[f'{strategy}_rougeL_f1'] = str(rouge_f1)
            #     variables[f'{strategy}_cosine_sim'] = str(cosine)

            # cot_score_path = f'{current_analysis}/cot_rouge_forget_score.json'
            # if os.path.exists(cot_score_path):
            #     cot_rougeL_f1 = extract_cot_rougeL_f1(cot_score_path)
            #     variables["cot_rougeL_f1"] = str(cot_rougeL_f1)
            # else:
            #     variables["cot_rougeL_f1"] = "NA"

            # # Step-level 계산 직접 수행
            # cosine_path = f'{current_analysis}/cot_rouge_forget_score_DefaultCoT.json'
            # if os.path.exists(cosine_path):
            #     with open(cosine_path, 'r') as f:
            #         lines = f.readlines()[:40]
            #     cosine_scores, rougeL_scores = [], []
            #     for line in lines:
            #         data = json.loads(line)
            #         steps1 = data['cot_answer'].split(". ")
            #         steps2 = data['generated_cot'].split(". ")
            #         emb1 = st_model.encode(steps1, convert_to_tensor=True)
            #         emb2 = st_model.encode(steps2, convert_to_tensor=True)
            #         cosine_matrix = util.pytorch_cos_sim(emb1, emb2)
            #         row_max_scores_cosine = cosine_matrix.max(dim=1).values.tolist()
            #         avg_cos = sum(row_max_scores_cosine) / len(row_max_scores_cosine)
            #         cosine_scores.append(avg_cos)
            #         rouge_max = []
            #         for ref in steps1:
            #             r_scores = [scorer.score(ref, hyp)['rougeL'].fmeasure for hyp in steps2]
            #             rouge_max.append(max(r_scores) if r_scores else 0.0)
            #         rougeL_scores.append(sum(rouge_max)/len(rouge_max))
            #     variables['cot_cosine_stepwise'] = str(round(sum(cosine_scores)/len(cosine_scores), 4))
            #     variables['cot_rougeL_f1_stepwise'] = str(round(sum(rougeL_scores)/len(rougeL_scores), 4))
            # else:
            #     variables['cot_cosine_stepwise'] = "NA"
            #     variables['cot_rougeL_f1_stepwise'] = "NA"

            # gpt_path = f'{current_analysis}/cot_gpteval_forget_score_DefaultCoT.json'
            # if os.path.exists(gpt_path):
            #     forgetting_score = extract_forgetting_score(gpt_path)
            #     variables['cot_forgetting_score'] = str(forgetting_score)
            # else:
            #     variables['cot_forgetting_score'] = "NA"

            # results_dict[current_method][current_epoch] = [
            #     variables['cot_rougeL_f1'],variables['cot_forgetting_score'],
            #     variables['cot_cosine_stepwise'],
            #     variables['cot_rougeL_f1_stepwise'],
            #     variables['DefaultCoT_rougeL_f1'], variables['DefaultCoT_cosine_sim'],
            #     variables['ZeroThink_rougeL_f1'], variables['ZeroThink_cosine_sim'],
            #     variables['LessThink_rougeL_f1'], variables['LessThink_cosine_sim'],
            # ] + [variables[key] for key in target_keys]
            results_dict[current_method][current_epoch] = [variables[key] for key in target_keys]

# -------------------- CSV 저장 --------------------
for method in methods:
    for epoch in sorted(results_dict[method].keys()):
        to_csv([method, str(epoch)] + results_dict[method][epoch], end_dir)