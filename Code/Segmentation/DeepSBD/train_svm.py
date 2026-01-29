import torch
import joblib
import numpy as np
from sklearn.svm import SVC
from torch.utils.data import DataLoader
from c3d_sbd import C3D_SBD
from sbd_dataset import LMDBVideoDataset

device = "cuda"

model = C3D_SBD().to(device)
model.load_state_dict(torch.load("weights_save/c3d_sbd_best.pth"))
model.eval()

print("chargement modèle ok")

dataset = LMDBVideoDataset("dataset/train_svm")  # idéalement différent du train CNN
loader = DataLoader(dataset, batch_size=1, shuffle=False)

print("chargement dataset ok")

X, y = [], []

with torch.no_grad():
    for x, label in loader:
        feat = model(x.to(device), return_features=True)
        feat = feat.cpu().numpy().squeeze()
        feat = feat / np.linalg.norm(feat)  # L2 norm

        X.append(feat)
        y.append(label.item())

svm = SVC(kernel="linear",probability=True)
svm.fit(X, y)

joblib.dump(svm, "weights_save/svm_deepsbd.pkl")
