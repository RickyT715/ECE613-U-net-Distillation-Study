import torch
import torch.nn as nn
import numpy as np
from torchmetrics.segmentation import DiceScore
from tqdm import tqdm
from distillation_loss import DistillationLoss
from torchmetrics import MetricCollection, F1Score, Accuracy, Precision, Specificity, Recall, JaccardIndex
import torchmetrics

best_result = {"epoch": -1, "teacher_acc": 0.0, "student_acc": 0.0}

def train_distillation(teacher_model, student_model, train_loader, val_loader, config, device):
    epochs = config["training"]["epochs"]
    lr = config["training"]["learning_rate"]
    alpha = config["distillation"]["alpha"]
    temperature = config["distillation"]["temperature"]
    save_path = config["distillation"]["save_path"]

    optimizer = torch.optim.Adam(student_model.parameters(), lr=lr)
    criterion = DistillationLoss(nn.CrossEntropyLoss(), temperature, alpha)

    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        student_model.train()
        losses = []
        print(f"\nTraining Epoch {epoch}/{epochs}")
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

        val_loss, teacher_acc, student_acc = validate(teacher_model, student_model, criterion, val_loader, device)

        if val_loss < best_loss:
            best_loss = val_loss
            best_result["epoch"] = epoch
            best_result["teacher_acc"] = teacher_acc
            best_result["student_acc"] = student_acc
            torch.save(student_model.state_dict(), save_path)
            print(f"Saved best student model at epoch {epoch} to {save_path}")

        print(f"[Epoch {epoch}] Train Loss: {np.mean(losses):.4f} | Val Loss: {val_loss:.4f} | Teacher Acc: {teacher_acc:.4f}, Student Acc: {student_acc:.4f}")

def validate(teacher_model, student_model, criterion, val_loader, device):
    teacher_model.eval()
    student_model.eval()

    evaluator_teacher = MetricCollection({
        "F1Score": torchmetrics.F1Score(num_classes=2, task="multiclass", average="macro"),
        "Accuracy": torchmetrics.Accuracy(num_classes=2, task="multiclass", average="macro"),
        "Dice": DiceScore(num_classes=2, average="macro"),
        "Precision": torchmetrics.Precision(num_classes=2, task="multiclass", average="macro"),
        "Specificity": torchmetrics.Specificity(num_classes=2, task="multiclass", average="macro"),
        "Recall": torchmetrics.Recall(num_classes=2, task="multiclass", average="macro"),
        "JaccardIndex": torchmetrics.JaccardIndex(num_classes=2, task="multiclass", average="macro"),
    }).to(device)

    evaluator_student = evaluator_teacher.clone().to(device)
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

            teacher_pred = torch.argmax(teacher_outputs, dim=1)
            student_pred = torch.argmax(student_outputs, dim=1)

            evaluator_teacher.update(teacher_pred, masks)
            evaluator_student.update(student_pred, masks)

    teacher_metrics = evaluator_teacher.compute()
    student_metrics = evaluator_student.compute()

    return np.mean(losses), float(teacher_metrics["Accuracy"]), float(student_metrics["Accuracy"])
