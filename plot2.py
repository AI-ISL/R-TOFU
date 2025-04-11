import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# 데이터 정의
raw_data = {
    "method": ["GA1+GD1"] * 5 + ["GA2+GD2"] * 5 + ["GA3+GD3"] * 5 + ["IDK1+GD1"] * 5 + ["IDK2+GD2"] * 5 + ["IDK3+GD3"] * 5,
    "epochs": [1, 2, 3, 4, 5] * 6,
    "Default": [
        0.5376, 0.5637, 0.5323, 0.4017, 0.3821,
        0.5217, 0.5505, 0.5201, 0.3775, 0.431,
        0.5723, 0.5834, 0.558, 0.4151, 0.4812,
        0.5254, 0.5366, 0.5317, 0.1584, 0.1468,
        0.5685, 0.5542, 0.5616, 0.0471, 0.0111,
        0.5722, 0.5679, 0.5336, 0.1942, 0.2058
    ],
    "ZeroThink": [
        0.4049, 0.4060, 0.4125, 0.3743, 0.3880,
        0.4157, 0.4296, 0.4072, 0.3709, 0.4017,
        0.4399, 0.4337, 0.4355, 0.374, 0.3924,
        0.4239, 0.418, 0.4138, 0.3433, 0.3705,
        0.4151, 0.4198, 0.3998, 0.0842, 0.0668,
        0.4334, 0.4428, 0.4384, 0.4042, 0.3997
    ],
    "LessThink": [
        0.4828, 0.4909, 0.4862, 0.4100, 0.4147,
        0.4848, 0.4755, 0.4777, 0.39, 0.4264,
        0.4899, 0.4868, 0.4748, 0.3868, 0.4512,
        0.4741, 0.4812, 0.4775, 0.3255, 0.3189,
        0.4861, 0.4947, 0.4689, 0.0894, 0.0699,
        0.4751, 0.4769, 0.4737, 0.4206, 0.4419
    ]
}

df = pd.DataFrame(raw_data)

# 각 그래프 개별 PDF로 저장
methods = df["method"].unique()
for method in methods:
    sub_df = df[df["method"] == method]
    filename = f"{method.replace('+', '_')}_think.pdf"
    with PdfPages(filename) as pdf:
        plt.figure(figsize=(8, 5))
        plt.plot(sub_df["epochs"], sub_df["Default"], marker='o', label='DefaultThink')
        plt.plot(sub_df["epochs"], sub_df["ZeroThink"], marker='s', label='ZeroThink')
        plt.plot(sub_df["epochs"], sub_df["LessThink"], marker='^', label='LessThink')
        plt.title(f'{method} - Think Strategies over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Score')
        plt.ylim(0, 0.6)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        pdf.savefig()
        plt.close()

print("저장완룐")
