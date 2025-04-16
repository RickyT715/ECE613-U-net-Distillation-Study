import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
from torch.utils.data import DataLoader
from torchmetrics.segmentation import DiceScore

from models.multiresunet import MultiResUnet
from models.student_unet_multiresunet import StudentUNet_MultiResUNet
# from datasets.isic import ISIC2018DatasetFast
from datasets.segpc import SegPC2021Dataset
from utils import load_config
from train_distillation.train_distillation_multiresunet import train_distillation, best_result
import torchmetrics
from torchmetrics import MetricCollection

def make_serializable(metrics):
    return {k: float(v.cpu().detach()) for k, v in metrics.items()}

def evaluate_metrics(model, dataloader, device):
    model.eval()
    evaluator = MetricCollection({
        "F1Score": torchmetrics.F1Score(num_classes=2, task="multiclass", average="macro"),
        "Accuracy": torchmetrics.Accuracy(num_classes=2, task="multiclass", average="macro"),
        "Dice": DiceScore(num_classes=2, task="multiclass", average="macro"),
        "Precision": torchmetrics.Precision(num_classes=2, task="multiclass", average="macro"),
        "Specificity": torchmetrics.Specificity(num_classes=2, task="multiclass", average="macro"),
        "Recall": torchmetrics.Recall(num_classes=2, task="multiclass", average="macro"),
        "JaccardIndex": torchmetrics.JaccardIndex(num_classes=2, task="multiclass", average="macro"),
    }).to(device)

    with torch.no_grad():
        for batch in dataloader:
            imgs = batch["image"].to(device)
            masks = batch["mask"].to(device)

            if masks.dim() == 4 and masks.size(1) > 1:
                masks = torch.argmax(masks, dim=1)
            elif masks.dim() == 4 and masks.size(1) == 1:
                masks = masks.squeeze(1)
            masks = masks.long()

            preds = model(imgs)
            preds = torch.argmax(preds, dim=1)
            evaluator.update(preds, masks)

    return make_serializable(evaluator.compute())

def main():
    config = load_config("../configs/distillation/multiresunet.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize teacher
    teacher = MultiResUnet(channels=4, filters=32, nclasses=2).to(device)
    teacher.load_state_dict(torch.load(config["distillation"]["teacher_model_path"], map_location=device))
    teacher.eval()

    # Initialize student
    student = StudentUNet_MultiResUNet(in_channels=4, out_channels=2).to(device)

    # tr_loader = DataLoader(ISIC2018DatasetFast(mode="tr", one_hot=False), **config["data_loader"]["train"])
    # vl_loader = DataLoader(ISIC2018DatasetFast(mode="vl", one_hot=False), **config["data_loader"]["validation"])

    tr_loader = DataLoader(SegPC2021Dataset(mode="tr", one_hot=False), **config["data_loader"]["train"])
    vl_loader = DataLoader(SegPC2021Dataset(mode="vl", one_hot=False), **config["data_loader"]["validation"])


    train_distillation(teacher, student, tr_loader, vl_loader, config, device)

    print("\nBest student model was saved.")
    print(f"Best epoch: {best_result['epoch']}")
    print(f"Student Accuracy: {best_result['student_acc']:.4f}")
    print(f"Teacher Accuracy: {best_result['teacher_acc']:.4f}")

    print("\nFinal Evaluation on Validation Set:")
    print("Teacher Metrics:")
    teacher_metrics = evaluate_metrics(teacher, vl_loader, device)
    for k, v in teacher_metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\nStudent Metrics:")
    student.load_state_dict(torch.load(config["distillation"]["save_path"], map_location=device))
    student.eval()
    student_metrics = evaluate_metrics(student, vl_loader, device)
    for k, v in student_metrics.items():
        print(f"  {k}: {v:.4f}")

if __name__ == "__main__":
    main()
