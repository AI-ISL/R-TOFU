import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import scienceplots

plt.style.use('science')

learning_rate = ['1e-05']
# methods = ['GA5+GD1', 'GA5+GD2', 'GA5+GD3','GA6+GD1', 'GA6+GD2', 'GA6+GD3','GA3+GD3']
# methods_label = ['GA5+GD1', 'GA5+GD2', 'GA5+GD3','GA6+GD1', 'GA6+GD2', 'GA6+GD3','GA3+GD3']
# methods = ['IDK1+GD1', 'IDK2+GD2', 'IDK3+GD3', 'IDK1+GD2', 'IDK1+GD3', 'IDK2+GD1', 'IDK2+GD3', 'IDK3+GD1', 'IDK3+GD2']
# methods_label = ['IDK1+GD1', 'IDK2+GD2', 'IDK3+GD3', 'IDK1+GD2', 'IDK1+GD3', 'IDK2+GD1', 'IDK2+GD3', 'IDK3+GD1', 'IDK3+GD2']
methods = ['IDK1+GD1', 'IDK2+GD2', 'IDK3+GD3', 'SDK+GD1', 'SDK+GD2', 'SDK+GD3']
methods_label = ['CoT+Answer', 'Answer','CoT', 'SDK+GD1', 'SDK+GD2', 'SDK+GD3']
def parse_single_results(lines):
    model_utility = None
    for line in lines:
        
        if 'Real Authors ROUGE:' in line:
            a1 = line.split(': ')[1].strip()
        if 'Real Authors Truth Ratio:' in line:
            a2 = line.split(': ')[1].strip()
        if 'Real Authors Cosine Similarity:' in line:
            a3 = line.split(': ')[1].strip()
        if 'Real Authors Entailment Score:' in line:
            a4 = line.split(': ')[1].strip()
        if 'Real Authors Token Entropy:' in line:
            a5 = line.split(': ')[1].strip()


        if 'Real World ROUGE:' in line:
            b1 = line.split(': ')[1].strip()
        if 'Real World Truth Ratio:' in line:
            b2 = line.split(': ')[1].strip()
        if 'Real World Cosine Similarity:' in line:
            b3 = line.split(': ')[1].strip()
        if 'Real World Entailment Score:' in line:
            b4 = line.split(': ')[1].strip()
        if 'Real World Token Entropy:' in line:
            b5 = line.split(': ')[1].strip()


        if 'Retain ROUGE:' in line:
            c1 = line.split(': ')[1].strip()
        if 'Retain Truth Ratio:' in line:
            c2 = line.split(': ')[1].strip()
        if 'Retain Cosine Similarity:' in line:
            c3 = line.split(': ')[1].strip()
        if 'Retain Entailment Score:' in line:
            c4 = line.split(': ')[1].strip()
        if 'Retain Token Entropy:' in line:
            c5 = line.split(': ')[1].strip()
        if 'Retain Probability:' in line:
            c6 = line.split(': ')[1].strip()


        if 'Forget ROUGE:' in line:
            d1 = line.split(': ')[1].strip()
        if 'Forget Truth Ratio:' in line:
            d2 = line.split(': ')[1].strip()
        if 'Forget Cosine Similarity:' in line:
            d3 = line.split(': ')[1].strip()
        if 'Forget Entailment Score:' in line:
            d4 = line.split(': ')[1].strip()
        if 'Forget Probability:' in line:
            d5 = line.split(': ')[1].strip()

        if 'Model Utility' in line:
            model_utility = line.split(': ')[1].strip()
        if 'Forget Efficacy' in line:
            forget_efficacy = line.split(': ')[1].strip()
        
    values = [float(a1), float(a2), float(a3), float(a4),
      float(b1), float(b2), float(b3), float(b4),
      float(c1), float(c2), float(c3), float(c4)]

    if all(v != 0 for v in values):
        MU = len(values) / sum(1.0 / v for v in values)
    else:
        MU = 0.0  # 또는 적절한 예외 처리

    FE = 1.0 - (float(d1) + float(d2) + float(d3) + float(d4)) / 4.0
    return MU, FE, float(model_utility), float(forget_efficacy)


top ="results/steps/llama3-8b/"
# (epoch, lr, method) -> (single, retain, forget)
results_dict = {}

# 중복 처리를 위해 visited set 사용
visited = set()

for root, dirs, files in os.walk(top):
    # 일단 epoch, lr 찾기
    found_epoch_lr = False
    current_epoch, current_lr = None, None
    
    for epoch in range(1, 3):
        for lr in learning_rate:
            if f'epoch{epoch}_{lr}_FixRef_maskTrue_1.0' in root or f'epoch{epoch}_{lr}_FixRef_maskFalse_1.0' in root:
                found_epoch_lr = True
                current_epoch = epoch
                current_lr = lr
                break
        if found_epoch_lr:
            break
    if not found_epoch_lr:
        continue
    
    # forget01 조건
    if 'forget01' not in root:
        continue
    
    # method 찾기
    found_method = False
    current_method = None
    for method in methods:
        if f'/{method}/' in root:
            found_method = True
            current_method = method
            break
    if not found_method:
        continue
    
    # 이미 방문했는지 체크 (epoch, lr, method 조합으로 중복 제거)
    key = (current_epoch, current_lr, current_method)
    if key in visited:
        continue

    # unlearn_times_1 폴더가 있는지
    if 'unlearn_times_1' in dirs:
        visited.add(key)  # 방문 처리

        current_analysis = f'{root}/unlearn_times_1/eval_results-last'
        unlearning_txt_dir = os.path.join(current_analysis, 'unlearning_results.txt')
        
        single_val = None
        if os.path.isfile(unlearning_txt_dir):
            with open(unlearning_txt_dir, 'r') as f:
                unlearning_lines = f.readlines()
                MU, FE, utility, efficacy = parse_single_results(unlearning_lines) 

        results_dict[key] = (MU, FE)


grouped_data = defaultdict(dict)
# grouped_data[(lr, method)][epoch] = (single_val, retain_val, forget_val)

for (epoch, lr, method), (utility, efficacy) in results_dict.items():
    grouped_data[(lr, method)][epoch] = (utility, efficacy)
    grouped_data[(lr, method)][0] = (0.6211, 0.2867)

colors = {
    "UGAD": "#d62728",  # Dark Red
    "MK": "#1f77b4",  # Deep Blue

}
plt.figure(figsize=(6, 4))
for i in range(len(methods)):
    for (lr, method), epoch_dict in grouped_data.items():
        epochs = sorted(epoch_dict.keys())
        utility_base = [epoch_dict[e][0] for e in epochs]
        efficacy_base = [epoch_dict[e][1] for e in epochs]


        if method == methods[i]:
            plt.plot(utility_base, efficacy_base,  marker='o', label=methods_label[i], markersize=12, linewidth=2)
            break



plt.xlabel("Model Utility", fontsize=20)
plt.ylabel("Forget Effifcacy", fontsize=20)
plt.legend(fontsize=10, loc='lower left', frameon=True)  # 범례 폰트 크기를 15로 증가

    # ✅ 오른쪽과 위쪽 눈금 제거
    # plt.tick_params(axis='both', which='both', bottom=True, left=True, right=False, top=False)
plt.tick_params(axis='both', which='major', labelsize=16, bottom=True, left=True, right=False, top=False)
plt.minorticks_off()  # ← 작은 눈금(minor ticks) 완전 제거
plt.minorticks_off()  # 작은 눈금 제거
# plt.xlim(0.5, 0.65)
# plt.ylim(0.3, 0.485)
# plt.xticks(np.linspace(0.5, 0.63, 6)) 
# plt.yticks(np.linspace(0.3, 0.6, 6))
plt.grid(True)
plt.axhline(y=0.4400, color='gray', linestyle='--', linewidth=1.5)
plt.axvline(x=0.5915, color='gray', linestyle='--', linewidth=1.5)
plt.tight_layout()
output_path = f"base_targeted_eval.pdf"
plt.savefig(output_path)
plt.show()