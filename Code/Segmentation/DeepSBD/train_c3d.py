# import torch
# from torch.utils.data import DataLoader
# from torch.amp import autocast, GradScaler
# from c3d_sbd import C3D_SBD
# from sbd_dataset import LMDBVideoDataset
# import multiprocessing

# torch.backends.cudnn.benchmark = True  # accélère le conv3D

# def train():
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print("Device:", device)

#     # ===============================
#     # Modèle + Optimizer
#     # ===============================
#     model = C3D_SBD().to(device)

#     for p in model.conv1.parameters():
#         p.requires_grad = False
#     for p in model.conv2.parameters():
#         p.requires_grad = False

#     optimizer = torch.optim.SGD(
#         filter(lambda p: p.requires_grad, model.parameters()),
#         lr=2e-4,
#         momentum=0.9
#     )

#     # ===============================
#     # Loss pondérée
#     # ===============================
#     counts = torch.tensor([32712, 7443, 2011], dtype=torch.float)
#     weights = 1.0 / counts
#     weights /= weights.sum()
#     criterion = torch.nn.CrossEntropyLoss(weight=weights.to(device))

#     # ===============================
#     # Dataset / DataLoader
#     # ===============================
#     train_set = LMDBVideoDataset("dataset/train_c3d")
#     print("Dataset size:", len(train_set))

#     batch_size = 1      
#     accum_steps = 2       

#     train_loader = DataLoader(
#         train_set,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=8,
#         pin_memory=True,
#         persistent_workers=True,
#     )

#     print("Chargement dataset OK")

#     # ===============================
#     # AMP
#     # ===============================
#     scaler = GradScaler()

#     # ===============================
#     # Training loop
#     # ===============================
#     model.train()
#     optimizer.zero_grad(set_to_none=True)

#     max_epochs = 20
#     patience = 3
#     min_delta = 1e-3

#     best_loss = float("inf")
#     epochs_no_improve = 0


#     for epoch in range(max_epochs):
#         running_loss = 0.0

#         for step, (x, y) in enumerate(train_loader):
#             x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

#             with autocast(device_type="cuda", dtype=torch.float16):
#                 logits = model(x)
#                 loss = criterion(logits, y)
#                 loss = loss / accum_steps  # gradient accumulation

#             scaler.scale(loss).backward()

#             if (step + 1) % accum_steps == 0:
#                 scaler.step(optimizer)
#                 scaler.update()
#                 optimizer.zero_grad(set_to_none=True)

#             running_loss += loss.item() * accum_steps

#         avg_loss = running_loss / len(train_loader)

#         print(f"Epoch {epoch} | loss={avg_loss:.4f}")

#         if avg_loss < best_loss - min_delta:
#             best_loss = avg_loss
#             epochs_no_improve = 0
#             torch.save(model.state_dict(), "c3d_sbd_best.pth")
#             print("✔ Nouveau meilleur modèle sauvegardé")
#         else:
#             epochs_no_improve += 1
#             print(f"No improvement ({epochs_no_improve}/{patience})")

#         if epochs_no_improve >= patience:
#             print("Early stopping déclenché")
#             break

#         torch.cuda.empty_cache()


#     # ===============================
#     # Sauvegarde
#     # ===============================
#     # torch.save(model.state_dict(), "c3d_sbd.pth")
#     # print("✔ Modèle sauvegardé")

# if __name__ == "__main__":
#     multiprocessing.freeze_support()  # obligatoire sur Windows
#     train()


import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from c3d_sbd import C3D_SBD
from sbd_dataset import LMDBVideoDataset
import multiprocessing
import os

torch.backends.cudnn.benchmark = True

def train(resume_ckpt="c3d_sbd_best.pth"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # ===============================
    # Modèle
    # ===============================
    model = C3D_SBD().to(device)

    # Freeze early layers
    for p in model.conv1.parameters():
        p.requires_grad = False
    for p in model.conv2.parameters():
        p.requires_grad = False

    # ===============================
    # Optimizer
    # ===============================
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=2e-4,
        momentum=0.9
    )

    # ===============================
    # LR Scheduler
    # ===============================
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,       # divise lr par 2 si plateau
        patience=2,       # patience en epochs
        min_lr=1e-6
    )

    # # ===============================
    # # Resume checkpoint (optionnel)
    # # ===============================
    # if resume_ckpt and os.path.exists(resume_ckpt):
    #     model.load_state_dict(torch.load(resume_ckpt, map_location=device))
    #     print(f"✔ Reprise depuis {resume_ckpt}")

    # ===============================
    # Loss pondérée
    # ===============================
    counts = torch.tensor([32712, 7443, 2011], dtype=torch.float)
    weights = 1.0 / counts
    weights /= weights.sum()
    criterion = torch.nn.CrossEntropyLoss(weight=weights.to(device))

    # ===============================
    # Datasets / DataLoaders
    # ===============================
    train_set = LMDBVideoDataset("dataset/train_c3d")
    val_set   = LMDBVideoDataset("dataset/val_c3d")

    batch_size = 1
    accum_steps = 2

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    # ===============================
    # AMP
    # ===============================
    scaler = GradScaler()

    # ===============================
    # Training params
    # ===============================
    max_epochs = 50  # plus d’epochs
    patience = 5     # early stopping
    min_delta = 1e-3

    best_val_loss = float("inf")
    epochs_no_improve = 0

    train_losses = []
    val_losses = []
    val_accuracies = []

    # ===============================
    # Training loop
    # ===============================
    for epoch in range(max_epochs):
        print(f"\n===== Epoch {epoch} =====")

        # -------- TRAIN --------
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0

        for step, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with autocast(device_type="cuda", dtype=torch.float16):
                logits = model(x)
                loss = criterion(logits, y)
                loss = loss / accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item() * accum_steps

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        print(f"Train loss: {train_loss:.4f}")

        # -------- VALIDATION --------
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                with autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(x)
                    loss = criterion(logits, y)

                val_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        val_loss /= len(val_loader)
        val_acc = correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"Val loss  : {val_loss:.4f}")
        print(f"Val acc   : {val_acc:.4f}")

        # -------- LR Scheduler --------
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Current LR: {current_lr:.2e}")


        # -------- EARLY STOPPING --------
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "c3d_sbd_best.pth")
            print("✔ Nouveau meilleur modèle sauvegardé")
        else:
            epochs_no_improve += 1
            print(f"No improvement ({epochs_no_improve}/{patience})")

        if epochs_no_improve >= patience:
            print("⛔ Early stopping déclenché")
            break

        torch.cuda.empty_cache()

    # ===============================
    # Graphiques de training
    # ===============================
    import matplotlib.pyplot as plt

    epochs_range = range(1, len(train_losses)+1)

    plt.figure()
    plt.plot(epochs_range, train_losses, label="Train loss")
    plt.plot(epochs_range, val_losses, label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training / Validation Loss")
    plt.legend()
    plt.grid(True)c
    plt.savefig("loss_curve.png", dpi=150)
    plt.close()

    plt.figure()
    plt.plot(epochs_range, val_accuracies, label="Val accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig("val_accuracy.png", dpi=150)
    plt.close()

    print("✔ Graphiques sauvegardés : loss_curve.png, val_accuracy.png")
    print("\n✔ Entraînement terminé")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    train()
