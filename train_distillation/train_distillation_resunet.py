import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from distillation_loss import DistillationLoss
import torchmetrics
from torchmetrics import MetricCollection
from torchmetrics.segmentation import DiceScore
import torchmetrics

best_result = {"epoch": -1, "val_loss": float("inf"), "accuracy": 0.0}

def train_distillation(teacher_model, student_model, train_loader, val_loader, config, device):
    epochs = config["training"]["epochs"]
    lr = config["training"]["learning_rate"]
    alpha = config["distillation"]["alpha"]
    temperature = config["distillation"]["temperature"]
    save_path = config["distillation"]["save_path"]

    optimizer = torch.optim.Adam(student_model.parameters(), lr=lr)
    criterion = DistillationLoss(nn.CrossEntropyLoss(), temperature, alpha)

    for epoch in range(1, epochs + 1):
        student_model.train()
        losses = []
        print(f"\n🚀 Training Epoch {epoch}/{epochs}")
        for batch in tqdm(train_loader):
            imgs = batch["image"].to(device)
            masks = batch["mask"].to(device)

            if masks.dim() == 4 and masks.size(1) > 1:
                masks = torch.argmax(masks, dim=1)
            masks = masks.long()

            with torch.no_grad():
                teacher_outputs = teacher_model(imgs)

            student_outputs = student_model(imgs)
            loss = criterion(student_outputs, teacher_outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        val_loss, acc = validate(teacher_model, student_model, criterion, val_loader, device)

        if val_loss < best_result["val_loss"]:
            best_result["val_loss"] = val_loss
            best_result["epoch"] = epoch
            best_result["accuracy"] = acc
            torch.save(student_model.state_dict(), save_path)
            print(f"✅ Saved best student model at epoch {epoch} to {save_path}")

        print(f"[Epoch {epoch}] Train Loss: {np.mean(losses):.4f} | Val Loss: {val_loss:.4f} | Student Accuracy: {acc:.4f}")

def validate(teacher_model, student_model, criterion, val_loader, device):
    student_model.eval()
    teacher_model.eval()

    evaluator = MetricCollection({
        "F1Score": torchmetrics.F1Score(num_classes=2, task="multiclass", average="macro"),
        "Accuracy": torchmetrics.Accuracy(num_classes=2, task="multiclass", average="macro"),
        "Dice": DiceScore(num_classes=2, average="macro"),
        "Precision": torchmetrics.Precision(num_classes=2, task="multiclass", average="macro"),
        "Specificity": torchmetrics.Specificity(num_classes=2, task="multiclass", average="macro"),
        "Recall": torchmetrics.Recall(num_classes=2, task="multiclass", average="macro"),
        "JaccardIndex": torchmetrics.JaccardIndex(num_classes=2, task="multiclass", average="macro"),
    }).to(device)

    losses = []

    with torch.no_grad():
        for batch in val_loader:
            imgs = batch["image"].to(device)
            masks = batch["mask"].to(device)

            if masks.dim() == 4 and masks.size(1) > 1:
                masks = torch.argmax(masks, dim=1)
            elif masks.dim() == 4 and masks.size(1) == 1:
                masks = masks.squeeze(1)
            masks = masks.long()

            teacher_outputs = teacher_model(imgs)
            student_outputs = student_model(imgs)

            loss = criterion(student_outputs, teacher_outputs, masks)
            losses.append(loss.item())

            preds = torch.argmax(student_outputs, dim=1)
            evaluator.update(preds, masks)

    metrics = evaluator.compute()
    return np.mean(losses), float(metrics["Accuracy"])
