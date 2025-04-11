import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def plot_methods(methods, base_dir="results/5"):
    """
    주어진 methods에 대하여,
    1~10 에폭의 unlearning_results.csv에서
    Model Utility와 Forget Efficacy를 추출하여
    개별 PDF 그래프로 저장합니다.
    
    methods: ["GA1+GD1", "GA2+GD2", ...] 형식의 리스트
    base_dir: CSV 파일들이 들어있는 최상위 디렉토리 (기본값 "results")
    """
    epochs = range(1, 6)  # 1 ~ 10 에폭
    
    for method in methods:
        utility_scores = []
        efficacy_scores = []

        # 에폭별로 CSV 읽어서 "Model Utility", "Forget Efficacy" 가져오기
        for epoch in epochs:
            csv_path = os.path.join(
                base_dir,
                "llama3-8b",
                "forget05",
                method,
                "seed_1001",
                f"epoch{epoch}_1e-05_FixRef_maskTrue_1.0_1.0",
                "1",
                "unlearn_times_1",
                "eval_results-last",
                "unlearning_results.csv"
            )
            if not os.path.isfile(csv_path):
                print(f"[WARN] CSV 파일을 찾을 수 없습니다: {csv_path}") 
                utility_scores.append(None)
                efficacy_scores.append(None)
                continue

            df = pd.read_csv(csv_path)
            # 필요 컬럼이 있는지 체크
            if "Model Utility" in df.columns and "Forget Efficacy" in df.columns:
                utility_scores.append(float(df["Model Utility"].iloc[-1]))
                efficacy_scores.append(float(df["Forget Efficacy"].iloc[-1]))
            else:
                print(f"[WARN] 필요한 컬럼이 존재하지 않습니다: {csv_path}")
                utility_scores.append(None)
                efficacy_scores.append(None)

        # 그래프 PDF 저장
        pdf_filename = f"5{method.replace('+', '_')}.pdf"
        with PdfPages(pdf_filename) as pdf:
            plt.figure(figsize=(8, 5))
            plt.plot(epochs, utility_scores, marker='o', label='Utility', color = 'orange')
            plt.plot(epochs, efficacy_scores, marker='^', label='Efficacy', color = 'green')
            plt.title(f'{method} Performance over Epochs')
            plt.xlabel('Epochs')
            plt.ylabel('Score')
            plt.ylim(0, 1)
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            pdf.savefig()
            plt.close()

        print(f"[INFO] '{method}' 결과가 '{pdf_filename}'에 저장되었습니다.")


if __name__ == "__main__":
    # 원하는 메소드를 리스트로 지정
    methods_to_plot = [
        "GA1+GD1",
        "GA2+GD2",
        "GA3+GD3",
        "GA1+KL1",
        "GA2+KL2",
        "GA3+KL3",
        "NPO1+GD1",
        "NPO2+GD2",
        "NPO3+GD3",
        "NPO1+KL1",
        "NPO2+KL2",
        "NPO3+KL3"
    ]
    
    # 함수 호출
    plot_methods(methods_to_plot)
    print("모든 그래프 저장 완료!")

# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_pdf import PdfPages

# # 에폭 정의
# epochs = list(range(1, 11))

# # 주어진 데이터 (결측값 처리 포함)
# data = {
#     "GA1+GD1": {
#         "CoT Efficacy": [0.3746, 0.3574, 0.3801, 0.4765, 0.4830, 0.4886, 0.5444, 0.7199, 0.8774, 0.9229],
#         "Utility": [0.6005, 0.6120, 0.6016, 0.5804, 0.5651, 0.5686, 0.5500, 0.5014, 0.1598, 0.0],
#         "Efficacy": [0.3848, 0.4106, 0.4183, 0.5106, 0.5357, 0.4914, 0.5603, 0.6564, 0.7943, 0.7974]
#     },
#     "GA2+GD2": {
#         "CoT Efficacy": [0.3232, 0.3373, 0.3295, 0.4035, 0.3905, 0.3745, 0.4625, 0.4699, 0.5153, 0.5652],
#         "Utility": [0.5966, 0.6071, 0.5948, 0.5637, 0.5650, 0.5629, 0.5520, 0.5254, 0.4511, 0.4199],
#         "Efficacy": [0.4132, 0.3927, 0.3829, 0.4360, 0.5032, 0.4676, 0.5129, 0.5574, 0.6537, 0.6879]
#     },
#     "GA3+GD3": {
#         "CoT Efficacy": [0.3424, 0.3561, 0.3621, 0.4802, 0.4243, 0.4813, 0.5608, 0.7391, 0.9368, 0.9466],
#         "Utility": [0.5906, 0.5940, 0.6003, 0.5206, 0.4952, 0.5219, 0.4777, 0.4187, 0.2109, 0.1135],
#         "Efficacy": [0.3906, 0.3735, 0.4032, 0.4889, 0.5245, 0.4704, 0.5382, 0.6973, 0.8041, 0.8118]
#     },
#     "IDK2+GD2": {
#         "CoT Efficacy": [0.3149, 0.3295, 0.3302, 0.3882, 0.6119, 0.3764, 0.4015, 0.4301, 0.4226, 0.4362],
#         "Utility": [0.6121, 0.6165, 0.6137, 0.4048, 0.4913, 0.4378, 0.3608, 0.2678, 0.1544, 0.0598],
#         "Efficacy": [0.3857, 0.3827, 0.4045, 0.7227, 0.7168, 0.7214, 0.7316, 0.7292, 0.7395, 0.7358]
#     },
#     "IDK1+GD1": {
#         "CoT Efficacy": [0.3471, 0.3710, 0.3704, 0.6359, 0.6268, 0.6210, 0.6209, 0.6848, 0.7146, 0.7155],
#         "Utility": [0.6136, 0.6113, 0.6102, 0.5274, 0.5248, 0.5116, 0.5215, 0.4759, 0.4187, 0.3233],
#         "Efficacy": [0.4029, 0.3716, 0.4162, 0.6310, 0.6132, 0.6017, 0.6190, 0.6886, 0.7206, 0.7330]
#     },
#     "IDK3+GD3": {
#         "CoT Efficacy": [0.3700, 0.3526, 0.3669, 0.6379, 0.6138, 0.6157, 0.6192, 0.6823, 0.7020, 0.7234],
#         "Utility": [0.6097, 0.6088, 0.6130, 0.5351, 0.5685, 0.5606, 0.5303, 0.1145, 0, 0],
#         "Efficacy": [0.4106, 0.4018, 0.4028, 0.5963, 0.5541, 0.5647, 0.6150, 0.7219, 0.7288, 0.7337]
#     },
#     "GA1+GD2": {
#         "CoT Efficacy": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         "Utility": [0.6138, 0.6054, 0.6066, 0.5880, 0.5772, 0.5765, 0.5583, 0.5231, 0.4077, 0.3210],
#         "Efficacy": [0.4154, 0.4050, 0.4020, 0.4576, 0.4666, 0.4918, 0.5152, 0.6633, 0.7372, 0.7866]
#     },
# }

# # 각 그래프를 개별 PDF 파일로 저장
# for method, values in data.items():
#     filename = f"{method.replace('+', '_')}.pdf"  # 파일명에서 '+'를 '_'로 대체
#     with PdfPages(filename) as pdf:
#         plt.figure(figsize=(8, 5))
#         plt.plot(epochs, values["CoT Efficacy"], marker='o', label='CoT Efficacy')
#         plt.plot(epochs, values["Utility"], marker='s', label='Utility')
#         plt.plot(epochs, values["Efficacy"], marker='^', label='Efficacy')
#         plt.title(f'{method} Performance over Epochs')
#         plt.xlabel('Epochs')
#         plt.ylabel('Score')
#         plt.ylim(0, 1)
#         plt.grid(True)
#         plt.legend()
#         plt.tight_layout()
#         pdf.savefig()
#         plt.close()

# print("저장완룐")
