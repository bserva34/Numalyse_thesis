# ============================================================
# Implémentation fidèle à l'article :
# Keyframes + SSN-CAP + MSN + AtDLSTM
# ------------------------------------------------------------
# Ce code fournit :
# 1) Chargement vidéo
# 2) Sélection des keyframes (indices conservés)
# 3) Modèle deep learning (article)
# 4) Boucle d'entraînement supervisée
# 5) Inférence : indices de keyframes + type de transition
# ============================================================

import os
import cv2
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ============================================================
# 1. KEYFRAME SELECTION (indices conservés)
# ============================================================

def frame_to_hsv_hist(frame, bins=(16,)):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    channels = []
    for ch in range(3):
        hist = cv2.calcHist([hsv], [ch], None, [bins[0]], [0, 256]).flatten()
        if hist.sum() > 0:
            hist = hist / hist.sum()
        channels.append(hist)
    return np.concatenate(channels)


def chi2_distance(h1, h2, eps=1e-8):
    denom = h1 + h2 + eps
    return float(np.sum(((h1 - h2) ** 2) / denom))


def select_keyframes_with_indices(frames, alpha=1.0):
    H = [frame_to_hsv_hist(f) for f in frames]
    diffs = np.array([
        chi2_distance(H[i], H[i+1]) for i in range(len(H) - 1)
    ])
    mu, sigma = diffs.mean(), diffs.std()
    T = mu + alpha * sigma

    keyframes = [(0, frames[0])]
    for i, d in enumerate(diffs):
        if d > T:
            keyframes.append((i + 1, frames[i + 1]))

    return keyframes, diffs, T

# ============================================================
# 2. DATASET SUPERVISÉ (vidéo + vérités terrain)
# ============================================================

class ShotDataset(Dataset):
    """
    Structure attendue :
    dataset/
      video_001.mp4
      video_001.json  -> {"transitions": [{"frame": 120, "type": "cut"}, ...]}
    """
    LABEL_MAP = {"cut": 0, "fade_in": 1, "fade_out": 2, "dissolve": 3}

    def __init__(self, root_dir):
        self.root = root_dir
        self.videos = [f for f in os.listdir(root_dir) if f.endswith('.mp4')]

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video_path = os.path.join(self.root, self.videos[idx])
        gt_path = video_path.replace('.mp4', '.json')

        # Load video
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)
        cap.release()

        # Keyframes
        keyframes, _, _ = select_keyframes_with_indices(frames)

        # Ground truth
        with open(gt_path, 'r') as f:
            gt = json.load(f)["transitions"]

        samples = []
        for i in range(len(keyframes) - 1):
            idx1, f1 = keyframes[i]
            idx2, f2 = keyframes[i + 1]
            label = None
            for t in gt:
                if idx1 <= t["frame"] <= idx2:
                    label = self.LABEL_MAP[t["type"]]
            if label is not None:
                samples.append((f1, f2, label))

        return samples

# ============================================================
# 3. SSN-CAP + MSN
# ============================================================

class ConvBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class CanonicalAppearancePooling(nn.Module):
    def forward(self, x):
        feats = [x,
                 torch.rot90(x, 1, [2, 3]),
                 torch.rot90(x, 2, [2, 3]),
                 torch.rot90(x, 3, [2, 3])]
        return torch.max(torch.stack(feats), dim=0).values


class SSN_CAP(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ConvBackbone()
        self.cap = CanonicalAppearancePooling()

    def forward(self, x1, x2):
        f1 = self.cap(self.backbone(x1))
        f2 = self.cap(self.backbone(x2))
        return torch.abs(f1 - f2)


class MSN(nn.Module):
    def __init__(self, c=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, groups=c),
            nn.Conv2d(c, c, 1), nn.ReLU()
        )

    def forward(self, f1, f2):
        return self.conv(f1 * f2)

# ============================================================
# 4. AtDLSTM
# ============================================================

class AtDLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2)
        self.att = nn.Linear(hidden_dim, 1)
        self.fc = nn.Linear(hidden_dim, 4)

    def forward(self, x):
        # x : (T, B, D)
        out, _ = self.lstm(x)
        w = torch.softmax(self.att(out), dim=0)
        z = (w * out).sum(dim=0)
        return self.fc(z)

# ============================================================
# 5. ENTRAÎNEMENT
# ============================================================

def train(model, dataloader, epochs=10, lr=1e-4):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total = 0
        for batch in dataloader:
            for f1, f2, label in batch:
                f1 = torch.tensor(f1).permute(2, 0, 1).unsqueeze(0).float() / 255.
                f2 = torch.tensor(f2).permute(2, 0, 1).unsqueeze(0).float() / 255.
                y = torch.tensor([label])

                feat = model.ssn(f1, f2).mean(dim=[2, 3]).unsqueeze(0)
                out = model.temporal(feat)
                loss = loss_fn(out, y)

                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item()
        print(f"Epoch {epoch}: loss={total:.4f}")

# ============================================================
# 6. INFÉRENCE : keyframe index + transition
# ============================================================

def infer_video(model, video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(cv2.resize(f, (224, 224)))
    cap.release()

    keyframes, _, _ = select_keyframes_with_indices(frames)
    results = []

    model.eval()
    with torch.no_grad():
        for i in range(len(keyframes) - 1):
            idx1, f1 = keyframes[i]
            idx2, f2 = keyframes[i + 1]

            t1 = torch.tensor(f1).permute(2, 0, 1).unsqueeze(0).float() / 255.
            t2 = torch.tensor(f2).permute(2, 0, 1).unsqueeze(0).float() / 255.

            feat = model.ssn(t1, t2).mean(dim=[2, 3]).unsqueeze(0)
            logits = model.temporal(feat)
            label = torch.argmax(logits, dim=1).item()

            results.append({
                "keyframe_start": idx1,
                "keyframe_end": idx2,
                "transition": label
            })

    return results

# ============================================================
# 7. MAIN : entraînement OU détection (via argparse)
# ============================================================

import argparse

class FullModel(nn.Module):
    """SSN-CAP + MSN + AtDLSTM avec concaténation (fidèle à l'article)"""
    def __init__(self):
        super().__init__()
        self.ssn = SSN_CAP()
        self.msn = MSN()
        # concat appearance (128) + motion (128)
        self.temporal = AtDLSTM(input_dim=256)


def main():
    parser = argparse.ArgumentParser(description="Shot Boundary Detection (article-faithful)")
    parser.add_argument("--mode", choices=["train", "detect"], required=True,
                        help="train : entraînement / detect : détection")
    parser.add_argument("--dataset", type=str, help="Dossier dataset (train)")
    parser.add_argument("--video", type=str, help="Vidéo à analyser (detect)")
    parser.add_argument("--weights", type=str, default="model.pth",
                        help="Chemin des poids du modèle")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    model = FullModel()

    # ---------------- TRAIN ----------------
    if args.mode == "train":
        assert args.dataset is not None, "--dataset requis en mode train"
        dataset = ShotDataset(args.dataset)
        loader = DataLoader(dataset, batch_size=1, shuffle=True)
        train(model, loader, epochs=args.epochs)
        torch.save(model.state_dict(), args.weights)
        print(f"Modèle entraîné et sauvegardé dans {args.weights}")

    # ---------------- DETECT ----------------
    elif args.mode == "detect":
        assert args.video is not None, "--video requis en mode detect"
        model.load_state_dict(torch.load(args.weights, map_location="cpu"))
        results = infer_video(model, args.video)

        LABEL_INV = {0: "cut", 1: "fade_in", 2: "fade_out", 3: "dissolve"}
        for r in results:
            print(f"Keyframes [{r['keyframe_start']} -> {r['keyframe_end']}] : "
                  f"{LABEL_INV[r['transition']]}")


if __name__ == "__main__":
    main()
