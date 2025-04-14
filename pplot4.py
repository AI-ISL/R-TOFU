import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import scienceplots

# plt.style.use('science')
# >>> "scienceplots"는 pip로 설치해야 하며, 없는 경우 스타일 설정 시 에러가 날 수 있습니다.
# >>> plt.style.use('science')  # 필요하다면 활성화

learning_rate = ['1e-05']
methods = ['GA1+GD1', 'GA2+GD2']
methods_label =  ['GA1+GD1', 'GA2+GD2']

def parse_single_results(lines):
    model_utility = None  
    for line in lines:
        
        if 'Real Authors ROUGE:' in line:
            a1 = line.split(': ')[1].strip()
        if 'Real Authors Probability' in line:
            a2 = line.split(': ')[1].strip()
        if 'Real Authors Cosine Similarity:' in line:
            a3 = line.split(': ')[1].strip()

        if 'Real World ROUGE:' in line:
            b1 = line.split(': ')[1].strip()
        if 'Real World Probability:' in line:
            b2 = line.split(': ')[1].strip()
        if 'Real World Cosine Similarity:' in line:
            b3 = line.split(': ')[1].strip()

        if 'Retain ROUGE:' in line:
            c1 = line.split(': ')[1].strip()
        if 'Retain Probability:' in line:
            c2 = line.split(': ')[1].strip()
        if 'Retain Cosine Similarity:' in line:
            c3 = line.split(': ')[1].strip()

        if 'Forget ROUGE:' in line:
            d1 = line.split(': ')[1].strip()
        if 'Forget Probability:' in line:
            d2 = line.split(': ')[1].strip()
        if 'Forget Cosine Similarity:' in line:
            d3 = line.split(': ')[1].strip()

        if 'Model Utility' in line:
            model_utility = line.split(': ')[1].strip()
        if 'Forget Efficacy' in line:
            forget_efficacy = line.split(': ')[1].strip()
        
    values = [float(a1), float(a2), float(a3),
              float(b1), float(b2), float(b3),
              float(c1), float(c2), float(c3)]

    if all(v != 0 for v in values):
        MU = len(values) / sum(1.0 / v for v in values)
    else:
        MU = 0.0  # 혹은 적절한 예외 처리

    FE = 1.0 - (float(d1) + float(d2) + float(d3)) / 3.0
    return MU, FE, float(model_utility), float(forget_efficacy)


top = "results/steps/llama3-8b/"

# (lr, method) -> { step: (MU, FE) }
results_dict = {}

# 중복 처리를 위해 visited set 사용
visited = set()

for root, dirs, files in os.walk(top):
    # (1) epoch=5만 체크 (고정)
    #     => root 경로 안에 "epoch5_..." 가 들어가야 함
    if not any(f"epoch5_{lr}_FixRef_maskTrue_1.0" in root or f"epoch5_{lr}_FixRef_maskFalse_1.0" in root for lr in learning_rate):
        continue

    # (2) forget01(혹은 forget05) 검사
    if 'forget01' not in root:
        continue
    
    # (3) method 찾기
    current_method = None
    for m in methods:
        if f'/{m}/' in root:
            current_method = m
            break
    if current_method is None:
        continue
    
    # (4) lr 찾기
    current_lr = None
    for lr in learning_rate:
        # epoch5_{lr}_FixRef_maskTrue_1.0 or maskFalse_1.0 둘 다 가능
        if f"epoch5_{lr}_FixRef_maskTrue_1.0" in root or f"epoch5_{lr}_FixRef_maskFalse_1.0" in root:
            current_lr = lr
            break
    if current_lr is None:
        continue

    # key = (lr, method)
    key = (current_lr, current_method)
    if key in visited:
        continue
    
    # (5) unlearn_times_1 폴더 확인
    if 'unlearn_times_1' in dirs:
        visited.add(key)  # 방문 처리

        # (6) eval_results-1 ~ eval_results-6 순회
        for step in range(1, 6):
            current_analysis = os.path.join(root, 'unlearn_times_1', f'eval_results-{step}')
            unlearning_txt_dir = os.path.join(current_analysis, 'unlearning_results.txt')
            
            if os.path.isfile(unlearning_txt_dir):
                with open(unlearning_txt_dir, 'r') as f:
                    unlearning_lines = f.readlines()
                    MU, FE, utility, efficacy = parse_single_results(unlearning_lines)
                
                # (7) 결과 저장 (step별)
                if key not in results_dict:
                    results_dict[key] = {}
                results_dict[key][step] = (MU, FE)
    # print(results_dict)


# 이제 results_dict는
#   {
#       (lr, method): {
#           1: (MU1, FE1),
#           2: (MU2, FE2),
#           ...
#           6: (MU6, FE6)
#       },
#       ...
#   }
# 이런 구조로 되어 있습니다.

########################################################################
# 플롯
########################################################################

plt.figure(figsize=(6, 4))

# methods와 methods_label이 1:1 대응
# - for i in range(len(methods)): ... 안에서
#   (lr, method)가 results_dict에 있으면 step1~6의 MU/FE를 그립니다.

for i in range(len(methods)):
    method_i = methods[i]
    label_i  = methods_label[i]

    # (lr, method) 중, 현재 method_i인 것만 찾는다.
    for (lr, m), step_dict in results_dict.items():
        if m == method_i:
            # step_dict = { 1: (MU1, FE1), 2: (MU2, FE2), ... }
            steps = sorted(step_dict.keys())  # [1, 2, 3, 4, 5, 6]
            utility_list = [step_dict[s][0] for s in steps]
            efficacy_list = [step_dict[s][1] for s in steps]

            plt.plot(utility_list, efficacy_list, marker='o',
                     label=label_i, markersize=12, linewidth=2)
            break  # 같은 method가 여러 번 찍히지 않도록 첫 번째만 그린 후 탈출

plt.xlabel("Model Utility", fontsize=20)
plt.ylabel("Forget Efficacy", fontsize=20)
plt.legend(fontsize=10, loc='lower left', frameon=True)

plt.tick_params(axis='both', which='major', labelsize=16,
                bottom=True, left=True, right=False, top=False)
plt.minorticks_off()  # 작은 눈금(minor ticks) 제거
plt.grid(True)

# 임의로 예시로 보이던 기준선
plt.axhline(y=0.56491285, color='gray', linestyle='--', linewidth=1.5)

plt.tight_layout()
output_path = "base_targeted_eval.pdf"
plt.savefig(output_path)
plt.show()
