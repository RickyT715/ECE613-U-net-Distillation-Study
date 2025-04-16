import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import torch
import numpy as np
from torch.utils.data import DataLoader
# from datasets.isic import ISIC2018DatasetFast
from datasets.segpc import SegPC2021Dataset
from models.student_unet_transunet import StudentTransUNet
from models.transunet.vit_seg_modeling import VisionTransformer, CONFIGS
from utils import load_config
from train_distillation.train_distillation_transunet import train_distillation, best_result
import torchmetrics
from torchmetrics import MetricCollection
from torchmetrics.segmentation import DiceScore


def make_serializeable_metrics(computed_metrics):
    res = {}
    for k, v in computed_metrics.items():
        res[k] = float(v.cpu().detach().numpy())
    return res


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
            elif masks.size(1) == 1:
                masks = masks.squeeze(1).long()

            preds = model(imgs)
            preds_ = torch.argmax(preds, dim=1)
            evaluator.update(preds_, masks)

    results = evaluator.compute()
    return make_serializeable_metrics(results)


def main():
    config = load_config("../configs/distillation/transunet.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #获取vision transformer
    vit_config = CONFIGS["R50-ViT-B_16"]
    vit_config.n_classes = config["model"]["params"]["out_channels"]
    vit_config.n_skip = 3
    img_size = config["model"]["params"]["img_size"]
    vit_config.patches.grid = (img_size // 16, img_size // 16)

    # vit_config = CONFIGS["R50-ViT-B_16"]
    # vit_config.n_classes = 2
    # vit_config.classifier = "seg"

    # vit_config.patches = ConfigDict()
    # vit_config.patches.grid = (14, 14)
    # vit_config.hybrid = True

    #初始化教师
    teacher = VisionTransformer(
        config=vit_config,
        img_size=img_size,
        num_classes=vit_config.n_classes
    ).to(device)


    #加载教师
    teacher_ckpt = torch.load(config["distillation"]["teacher_model_path"], map_location=device)
    teacher.load_state_dict(teacher_ckpt)
    teacher.eval()

    #初始学生
    student = StudentTransUNet(
        in_channels=config["model"]["params"]["in_channels"],
        out_channels=config["model"]["params"]["out_channels"]
    ).to(device)

    #加载数据
    # tr_ds = ISIC2018DatasetFast(mode="tr", one_hot=False)
    # vl_ds = ISIC2018DatasetFast(mode="vl", one_hot=False)

    tr_ds = SegPC2021Dataset(mode="tr", one_hot=False)
    vl_ds = SegPC2021Dataset(mode="vl", one_hot=False)

    tr_loader = DataLoader(tr_ds, **config["data_loader"]["train"])
    vl_loader = DataLoader(vl_ds, **config["data_loader"]["validation"])

    #训练
    train_distillation(teacher, student, tr_loader, vl_loader, config, device)

    # 训练表现
    print("\n✅ Best student model was saved.")
    print(f"Best epoch: {best_result['epoch']}")
    print(f"Student Accuracy: {best_result['student_acc']:.4f}")
    print(f"Teacher Accuracy: {best_result['teacher_acc']:.4f}")

    #验证
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
