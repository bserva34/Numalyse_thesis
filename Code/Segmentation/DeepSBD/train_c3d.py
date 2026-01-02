import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from c3d_sbd import C3D_SBD
from sbd_dataset import LMDBVideoDataset
import multiprocessing

torch.backends.cudnn.benchmark = True  # accélère le conv3D

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # ===============================
    # Modèle + Optimizer
    # ===============================
    model = C3D_SBD().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=2e-4, momentum=0.9)

    # ===============================
    # Loss pondérée
    # ===============================
    counts = torch.tensor([32712, 7443, 2011], dtype=torch.float)
    weights = 1.0 / counts
    weights /= weights.sum()
    criterion = torch.nn.CrossEntropyLoss(weight=weights.to(device))

    # ===============================
    # Dataset / DataLoader
    # ===============================
    train_set = LMDBVideoDataset("dataset/train")
    print("Dataset size:", len(train_set))

    batch_size = 2        # batch réel = 2
    accum_steps = 4       # gradient accumulation pour batch effectif = 8

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )

    print("Chargement dataset OK")

    # ===============================
    # AMP
    # ===============================
    scaler = GradScaler()

    # ===============================
    # Training loop
    # ===============================
    model.train()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(3):
        running_loss = 0.0

        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            with autocast(device_type="cuda"):
                logits = model(x)
                loss = criterion(logits, y)
                loss = loss / accum_steps  # gradient accumulation

            scaler.scale(loss).backward()

            if (step + 1) % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item() * accum_steps

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch} | loss={avg_loss:.4f}")

        torch.cuda.empty_cache()


    # ===============================
    # Sauvegarde
    # ===============================
    torch.save(model.state_dict(), "c3d_sbd.pth")
    print("✔ Modèle sauvegardé")

if __name__ == "__main__":
    multiprocessing.freeze_support()  # obligatoire sur Windows
    train()
