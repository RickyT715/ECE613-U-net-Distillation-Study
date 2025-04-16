import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchmetrics import MetricCollection, F1Score, Accuracy, Precision, Specificity, Recall, JaccardIndex
from torchmetrics.segmentation import DiceScore
from models.student_unet_attunet import StudentUNet_AttUNet
from models.attunet import AttU_Net
from models.student_unet_missformer import StudentMissFormer
from models.missformer.MISSFormer import MISSFormer
from models.multiresunet import MultiResUnet
from models.student_unet_multiresunet import StudentUNet_MultiResUNet
from models.resunet.res_unet import ResUnet
from models.student_unet_resunet import StudentUNet_ResUNet
from models.student_unet_transunet import StudentTransUNet
from models.transunet.vit_seg_modeling import VisionTransformer, CONFIGS
from models.student_unet_uctransnet import StudentUCTransUNet
from models.uctransnet.UCTransNet import UCTransNet
from models.uctransnet.Config import get_CTranS_config
from models.student_unet_unet import StudentUNet
from models.unet import UNet
from models.unetpp import NestedUNet
from models.student_unet_unetpp import StudentUNet_UNetPP
from torchvision.utils import make_grid
from PIL import Image, ImageDraw, ImageFont

# from datasets.isic import ISIC2018DatasetFast
from datasets.segpc import SegPC2021Dataset


from ml_collections import ConfigDict
from models.transunet.vit_seg_modeling import VisionTransformer, CONFIGS

#build teacher helper
def build_transunet_teacher():
    config = CONFIGS["R50-ViT-B_16"]
    config.n_classes = 2
    config.classifier = "seg"

    config.patches = ConfigDict()
    config.patches.grid = (14, 14)
    config.hybrid = True

    return VisionTransformer(config, img_size=224, num_classes=2)

model_info = [
    ("unet", lambda: UNet(4, 2), lambda: StudentUNet(4, 2)),
    ("attunet", lambda: AttU_Net(img_ch=4, output_ch=2), lambda: StudentUNet_AttUNet(4, 2)),
    ("unetpp", lambda: NestedUNet(num_classes=2, input_channels=4), lambda: StudentUNet_UNetPP(4, 2)),
    ("multiresunet", lambda: MultiResUnet(channels=4, filters=32, nclasses=2), lambda: StudentUNet_MultiResUNet(4, 2)),
    ("resunet", lambda: ResUnet(4, 2), lambda: StudentUNet_ResUNet(4, 2)),
    ("transunet", lambda: build_transunet_teacher(), lambda: StudentTransUNet(4, 2)),
    ("uctransnet", lambda: UCTransNet(get_CTranS_config(), n_channels=4, n_classes=2, img_size=224), lambda: StudentUCTransUNet(4, 2)),
    ("missformer", lambda: MISSFormer(in_ch=4, num_classes=2), lambda: StudentMissFormer(4, 2)),
]

def evaluate(model, dataloader, device):
    model.eval()
    metrics = MetricCollection({
        "F1Score": F1Score(num_classes=2, task="binary", average="macro"),
        "Accuracy": Accuracy(num_classes=2, task="binary"),
        "Dice": DiceScore(num_classes=2, average="macro"),
        "Precision": Precision(num_classes=2, task="binary", average="macro"),
        "Specificity": Specificity(num_classes=2, task="binary", average="macro"),
        "Recall": Recall(num_classes=2, task="binary", average="macro"),
        "JaccardIndex": JaccardIndex(num_classes=2, task="binary", average="macro"),
    }).to(device)

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            if masks.dim() == 4 and masks.size(1) > 1:
                masks = torch.argmax(masks, dim=1)
            elif masks.dim() == 4 and masks.size(1) == 1:
                masks = masks.squeeze(1)
            masks = masks.long()

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            metrics.update(preds, masks)

    return {k: float(v.cpu()) for k, v in metrics.compute().items()}

#单个图片segmentation，应该用不上了
def visualize_segmentation(image, mask, pred, path):
    image_np = image.cpu().numpy().transpose(1, 2, 0)
    mask_np = mask.cpu().numpy()
    pred_np = pred.cpu().numpy()
    overlay = np.zeros_like(image_np)
    for i in np.unique(mask_np):
        if i == 0: continue
        overlay[mask_np == i] = np.random.randint(0, 255, 3)
    blended = 0.5 * image_np + 0.5 * overlay / 255.0
    plt.imsave(path, blended)

#对比图生成
def visualize_comparison(image, mask_teacher, pred_teacher, pred_student, path):
    image_np = image.cpu().numpy().transpose(1, 2, 0)
    
    if image_np.shape[2] == 4:
        image_np = image_np[:, :, :3]

    teacher_overlay = np.zeros_like(image_np)
    student_overlay = np.zeros_like(image_np)

    for i in np.unique(pred_teacher.cpu().numpy()):
        if i == 0: continue
        teacher_overlay[pred_teacher.cpu().numpy() == i] = np.random.randint(0, 255, 3)
    for i in np.unique(pred_student.cpu().numpy()):
        if i == 0: continue
        student_overlay[pred_student.cpu().numpy() == i] = np.random.randint(0, 255, 3)

    blended_teacher = (0.5 * image_np + 0.5 * teacher_overlay / 255.0)
    blended_student = (0.5 * image_np + 0.5 * student_overlay / 255.0)

    concat = np.concatenate([blended_teacher, blended_student], axis=1)
    concat = (concat * 255).astype(np.uint8)

    concat_pil = Image.fromarray(concat)
    draw = ImageDraw.Draw(concat_pil)

    try:
        font = ImageFont.truetype("arial.ttf", size=24)
    except:
        font = ImageFont.load_default()

    draw.text((10, 10), "Teacher", fill=(255, 255, 255), font=font)
    draw.text((concat.shape[1] // 2 + 10, 10), "Student", fill=(255, 255, 255), font=font)

    concat_pil.save(path)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vl_ds = SegPC2021Dataset(mode="vl", one_hot=False)
    val_loader = DataLoader(vl_ds, batch_size=1, shuffle=False)

    results = []
    os.makedirs("comparisons", exist_ok=True)

    for name, build_teacher, build_student in model_info:
        print(f"Evaluating {name}...")

        teacher_model = build_teacher().to(device)
        student_model = build_student().to(device)

        teacher_path = f"saved_models/teacher/segpc2021_{name}/best_model_state_dict.pt"
        student_path = f"saved_models/student/segpc/{name}_student.pth"

        teacher_model.load_state_dict(torch.load(teacher_path, map_location=device))
        student_model.load_state_dict(torch.load(student_path, map_location=device))

        teacher_metrics = evaluate(teacher_model, val_loader, device)
        student_metrics = evaluate(student_model, val_loader, device)

        teacher_model.eval()
        student_model.eval()
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= 8:
                    break
                image = batch["image"].to(device)[0]
                mask = batch["mask"].to(device)[0]
                if mask.dim() == 3:
                    if mask.size(0) > 1:
                        mask = torch.argmax(mask, dim=0)
                    else:
                        mask = mask.squeeze(0)
                pred_teacher = torch.argmax(teacher_model(image.unsqueeze(0)), dim=1)[0]
                pred_student = torch.argmax(student_model(image.unsqueeze(0)), dim=1)[0]

                vis_path = f"comparisons/{name}_compare_{i}.png"
                visualize_comparison(image, mask, pred_teacher, pred_student, vis_path)

        results.append({"Model": name.capitalize() + " Teacher", **teacher_metrics})
        results.append({"Model": name.capitalize() + " Student", **student_metrics})

    df = pd.DataFrame(results)
    df.to_csv("distillation_full_validation.csv", index=False)

    #生成简报
    # with open("distillation_report.txt", "w") as f:
    #     for row in results:
    #         f.write(f"Model: {row['Model']}\n")
    #         for key in ["Accuracy", "Dice", "F1Score", "JaccardIndex", "Precision", "Recall", "Specificity"]:
    #             f.write(f"  {key}: {row[key]:.6f}\n")
    #         f.write("\n")

    print("All good")

