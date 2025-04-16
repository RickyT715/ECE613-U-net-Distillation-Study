import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import torch
import numpy as np
from torch.utils.data import DataLoader
# from datasets.isic import ISIC2018DatasetFast
from datasets.segpc import SegPC2021Dataset
from models.student_unet_uctransnet import StudentUCTransUNet
from models.uctransnet.UCTransNet import UCTransNet
from models.uctransnet.Config import get_CTranS_config
from utils import load_config
import torchmetrics
from train_distillation.train_distillation_uctransnet import train_distillation, best_result
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy, Precision, Recall, Specificity, F1Score, JaccardIndex
from torchmetrics.segmentation import DiceScore


def make_serializeable_metrics(computed_metrics):
    res = {}
    for k, v in computed_metrics.items():
        res[k] = float(v.cpu().detach().numpy())
    return res


def teacher_validate_metrics(model, dataloader, device):
    model.eval()
    evaluator = MetricCollection({
        "F1Score": torchmetrics.F1Score(num_classes=2, task="multiclass", average="macro"),
        "Accuracy": torchmetrics.Accuracy(num_classes=2, task="multiclass", average="macro"),
        "Dice": DiceScore(num_classes=2, average="macro"),
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
            elif masks.size(1) == 1:
                masks = masks.squeeze(1).long()

            preds = model(imgs)
            preds_ = torch.argmax(preds, dim=1)

            evaluator.update(preds_, masks)

    results = evaluator.compute()
    return make_serializeable_metrics(results)


def main():
    config = load_config("../configs/distillation/uctransnet.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #初始化教师
    uct_config = get_CTranS_config()
    uct_config.n_classes = config["model"]["params"]["out_channels"]
    teacher = UCTransNet(
        uct_config,
        n_channels=config["model"]["params"]["in_channels"],
        n_classes=config["model"]["params"]["out_channels"],
        img_size=config["model"]["params"]["img_size"]
    ).to(device)

    #加载教师
    teacher_ckpt = torch.load(config["distillation"]["teacher_model_path"], map_location=device)
    teacher.load_state_dict(teacher_ckpt)
    teacher.eval()

    #初始化学生
    student = StudentUCTransUNet(in_channels=4, out_channels=2).to(device)

    #数据加载
    # tr_ds = ISIC2018DatasetFast(mode="tr", one_hot=False)
    # vl_ds = ISIC2018DatasetFast(mode="vl", one_hot=False)

    tr_ds = SegPC2021Dataset(mode="tr", one_hot=False)
    vl_ds = SegPC2021Dataset(mode="vl", one_hot=False)

    tr_loader = DataLoader(tr_ds, **config["data_loader"]["train"])
    vl_loader = DataLoader(vl_ds, **config["data_loader"]["validation"])

    #训练
    train_distillation(teacher, student, tr_loader, vl_loader, config, device)

    print("\n✅ Best student model was saved.")
    print(f"Best epoch: {best_result['epoch']}")
    print(f"Student Accuracy: {best_result['student_acc']:.4f}")
    print(f"Teacher Accuracy: {best_result['teacher_acc']:.4f}")
    print("\n🎯 Final Evaluation on Validation Set:")

    print("Teacher Metrics:")
    teacher_metrics = teacher_validate_metrics(teacher, vl_loader, device)
    for k, v in teacher_metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\nStudent Metrics:")
    student.load_state_dict(torch.load(config["distillation"]["save_path"], map_location=device))
    student.eval()
    student_metrics = teacher_validate_metrics(student, vl_loader, device)
    for k, v in student_metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
