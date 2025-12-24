import torch
import joblib
from sklearn.svm import SVC
from torch.utils.data import DataLoader
from models.c3d_sbd import C3D_SBD
from dataset.sbd_dataset import SBDDataset

device = "cuda"
model = C3D_SBD().to(device)
model.load_state_dict(torch.load("c3d_sbd.pth"))
model.eval()

dataset = SBDDataset("dataset/train")
loader = DataLoader(dataset, batch_size=1)

X, y = [], []

with torch.no_grad():
    for x, label in loader:
        feat = model(x.to(device), return_features=True)
        X.append(feat.cpu().numpy().squeeze())
        y.append(label.item())

svm = SVC(kernel="linear", probability=True)
svm.fit(X, y)

joblib.dump(svm, "svm_deepsbd.pkl")
