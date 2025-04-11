import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# 데이터 정의
epochs = [1, 2, 3, 4, 5]

data = {
    "GA1+GD1": {
        "CoT Efficacy": [0.3746, 0.3574, 0.3801, 0.4765, 0.4830],
        "Utility": [0.6005, 0.6120, 0.6016, 0.5804, 0.5651],
        "Efficacy": [0.3848, 0.4106, 0.4183, 0.5106, 0.5357]
    },
    "GA2+GD2": {
        "CoT Efficacy": [0.3232, 0.3373, 0.3295, 0.4035, 0.3905],
        "Utility": [0.5966, 0.6071, 0.5948, 0.5637, 0.5650],
        "Efficacy": [0.4132, 0.3927, 0.3829, 0.4360, 0.5032]
    },
    "GA3+GD3": {
        "CoT Efficacy": [0.3424, 0.3561, 0.3621, 0.4802, 0.4243],
        "Utility": [0.5906, 0.5940, 0.6003, 0.5206, 0.4952],
        "Efficacy": [0.3906, 0.3735, 0.4032, 0.4889, 0.5245]
    },
    "IDK1+GD1": {
        "CoT Efficacy": [0.3471, 0.3710, 0.3704, 0.6359, 0.6268],
        "Utility": [0.6136, 0.6113, 0.6102, 0.5274, 0.5248],
        "Efficacy": [0.4029, 0.3716, 0.4162, 0.6310, 0.6132]
    },
    "IDK3+GD3": {
        "CoT Efficacy": [0.3700, 0.3526, 0.3669, 0.6379, 0.6138],
        "Utility": [0.6097, 0.6088, 0.6130, 0.5351, 0.5685],
        "Efficacy": [0.4106, 0.4018, 0.4028, 0.5963, 0.5541]
    },
    "IDK2+GD2": {
        "CoT Efficacy": [0.3149, 0.3295, 0.3302, 0.3882, 0.6119],
        "Utility": [0.6121, 0.6165, 0.6137, 0.4048, 0.4913],
        "Efficacy": [0.3857, 0.3827, 0.4045, 0.7227, 0.7168]
    }
}
# 각 그래프를 개별 PDF 파일로 저장
for method, values in data.items():
    filename = f"{method.replace('+', '_')}.pdf"  # 파일명에서 '+'를 '_'로 대체
    with PdfPages(filename) as pdf:
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, values["CoT Efficacy"], marker='o', label='CoT Efficacy')
        plt.plot(epochs, values["Utility"], marker='s', label='Utility')
        plt.plot(epochs, values["Efficacy"], marker='^', label='Efficacy')
        plt.title(f'{method} Performance over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Score')
        plt.ylim(0, 0.8)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        pdf.savefig()
        plt.close()

print("저장완룐")