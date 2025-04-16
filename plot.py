import pandas as pd
import matplotlib.pyplot as plt
import os

# 创建保存目录
os.makedirs("plots", exist_ok=True)

# 读取验证结果
df = pd.read_csv("distillation_full_validation.csv")
models = sorted(set(m.split()[0] for m in df["Model"]))

# 所有指标列名
metrics = ["Accuracy", "Dice", "F1Score", "JaccardIndex", "Precision", "Recall", "Specificity"]

for metric in metrics:
    teacher_vals = []
    student_vals = []
    drops = []

    for model in models:
        t_val = df[df["Model"] == f"{model} Teacher"][metric].values[0]
        s_val = df[df["Model"] == f"{model} Student"][metric].values[0]
        teacher_vals.append(round(t_val, 6))
        student_vals.append(round(s_val, 6))
        drops.append(round(t_val - s_val, 6))

    # 图1：Teacher vs Student 折线图
    plt.figure(figsize=(10, 5))
    plt.plot(models, teacher_vals, marker="o", label="Teacher", linewidth=2)
    plt.plot(models, student_vals, marker="o", label="Student", linewidth=2)
    plt.title(f"Teacher vs Student - {metric}")
    plt.ylabel(metric)
    plt.xlabel("Model")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"plots/{metric.lower()}_comparison.png")
    plt.close()

    # 图2：下降柱状图
    plt.figure(figsize=(10, 5))
    plt.bar(models, drops, color='orange')
    plt.title(f"{metric} Drop")
    plt.ylabel(f"{metric} Drop")
    plt.xlabel("Model")
    plt.grid(axis="y")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"plots/{metric.lower()}_drop.png")
    plt.close()

print("✅ 所有图已保存到 ./plots 文件夹")
