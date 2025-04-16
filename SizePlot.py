import matplotlib.pyplot as plt

model_sizes = {
    "UNet":       {"teacher": 7617, "student": 543},
    "AttUNet":    {"teacher": 136385, "student": 119},
    "UNet++":     {"teacher": 35891, "student": 227},
    "MultiResUNet": {"teacher": 30734, "student": 228},
    "ResUNet":    {"teacher": 51039, "student": 114},
    "TransUNet":  {"teacher": 411419, "student": 1834},
    "UCTransNet": {"teacher": 259695, "student": 1834},
    "MISSFormer": {"teacher": 166133, "student": 1834},
}

labels = list(model_sizes.keys())
teacher_sizes = [model_sizes[k]["teacher"] / 1024 for k in labels]
student_sizes = [model_sizes[k]["student"] / 1024 for k in labels]
size_ratios = [s / t * 100 for s, t in zip(student_sizes, teacher_sizes)]

x = range(len(labels))
width = 0.35

plt.figure(figsize=(12, 6))
plt.bar(x, teacher_sizes, width, label='Teacher Size (MB)')
plt.bar([i + width for i in x], student_sizes, width, label='Student Size (MB)')

for i, (t, s, r) in enumerate(zip(teacher_sizes, student_sizes, size_ratios)):
    plt.text(i + width, s + 1, f"{r:.1f}%", ha='center', va='bottom', fontsize=8)

plt.xticks([i + width / 2 for i in x], labels)
plt.ylabel("Model Size (MB)")
plt.title("Model Size Comparison: Teacher vs. Student")
plt.legend()
plt.tight_layout()
plt.grid(axis='y')
plt.show()
