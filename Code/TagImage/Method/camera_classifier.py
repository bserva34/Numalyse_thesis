import torch
import torchvision
import pytorch_lightning as pl

from PIL import Image
from torchvision import transforms
import cv2


# =========================================================
# LABELS
# =========================================================

ANGLE_LABELS = [
    "dutch",
    "high",
    "low",
    "neutral",
    "overhead"
]

LEVEL_LABELS = [
    "aerial",
    "eye",
    "ground",
    "hip",
    "knee",
    "shoulder"
]


# =========================================================
# DEVICE
# =========================================================

device = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================
# MODEL
# =========================================================

class ResNet(pl.LightningModule):

    def __init__(self, num_angle_classes=5, num_level_classes=6):
        super().__init__()

        self.resnet = torchvision.models.resnet18(pretrained=True)

        self.resnet.fc = torch.nn.Linear(
            self.resnet.fc.in_features,
            512
        )

        self.angle_head = torch.nn.Linear(
            512,
            num_angle_classes
        )

        self.level_head = torch.nn.Linear(
            512,
            num_level_classes
        )

    def forward(self, x):

        x = self.resnet(x)

        angle_logits = self.angle_head(x)
        level_logits = self.level_head(x)

        return angle_logits, level_logits


# =========================================================
# LOAD MODEL
# =========================================================

def load_model(checkpoint_path="model.ckpt"):

    model = ResNet.load_from_checkpoint(
        checkpoint_path,
        num_angle_classes=5,
        num_level_classes=6
    )

    model = model.to(device)

    model.eval()

    return model


# =========================================================
# TRANSFORM
# =========================================================

imnet_mean = [0.485, 0.456, 0.406]
imnet_std = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=imnet_mean,
        std=imnet_std
    )
])


# =========================================================
# SINGLE FRAME PREDICTION
# =========================================================

def predict_frame(model, frame_bgr):

    # OpenCV BGR -> RGB
    frame_rgb = cv2.cvtColor(
        frame_bgr,
        cv2.COLOR_BGR2RGB
    )

    pil = Image.fromarray(frame_rgb)

    x = transform(pil).unsqueeze(0).to(device)

    with torch.no_grad():

        angle_logits, level_logits = model(x)

    angle_pred = torch.argmax(
        angle_logits,
        dim=1
    ).item()

    level_pred = torch.argmax(
        level_logits,
        dim=1
    ).item()

    return {
        "angle_id": angle_pred,
        "angle": ANGLE_LABELS[angle_pred],

        "level_id": level_pred,
        "level": LEVEL_LABELS[level_pred]
    }


# =========================================================
# BATCH PREDICTION
# =========================================================

def predict_batch(model, frames_bgr):

    tensors = []

    # preprocessing
    for frame in frames_bgr:

        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        pil = Image.fromarray(frame_rgb)

        tensor = transform(pil)

        tensors.append(tensor)

    batch = torch.stack(tensors).to(device)

    # inference
    with torch.no_grad():

        angle_logits, level_logits = model(batch)

    angle_preds = torch.argmax(
        angle_logits,
        dim=1
    ).cpu().numpy()

    level_preds = torch.argmax(
        level_logits,
        dim=1
    ).cpu().numpy()

    angle_probs = torch.softmax(
        angle_logits,
        dim=1
    ).cpu().numpy()

    level_probs = torch.softmax(
        level_logits,
        dim=1
    ).cpu().numpy()



    results = []

    for angle_pred, level_pred, angle_prob, level_prob in zip(
        angle_preds,
        level_preds,
        angle_probs,
        level_probs
    ):
        results.append({

            "angle_id": int(angle_pred),
            "angle": ANGLE_LABELS[angle_pred],

            "level_id": int(level_pred),
            "level": LEVEL_LABELS[level_pred],

            "angle_probs": angle_prob,
            "level_probs": level_prob
        })

    return results
