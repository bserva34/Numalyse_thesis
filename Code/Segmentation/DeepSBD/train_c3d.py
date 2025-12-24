import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from c3d_sbd import C3D_SBD
from sbd_dataset import LMDBVideoDataset
import multiprocessing

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # modèle et optimizer
    model = C3D_SBD().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

    # pondération de la loss
    counts = torch.tensor([32712, 7443, 2011], dtype=torch.float)
    weights = 1.0 / counts
    weights = weights / weights.sum()
    criterion = torch.nn.CrossEntropyLoss(weight=weights.to(device))

    # dataset
    train_set = LMDBVideoDataset("dataset/train")

    print("Dataset size:", len(train_set))


    # oversampling
    # sample_weights = [weights[label] for _, label in train_set]
    # sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    # # DataLoader optimisé pour Windows
    # train_loader = DataLoader(
    #     train_set,
    #     batch_size=20,
    #     sampler=sampler,
    #     num_workers=16,       
    #     pin_memory=True
    # )

    # sans oversampling
    train_loader = DataLoader(train_set, batch_size=20,shuffle=True,num_workers=4,pin_memory=True,persistent_workers=True)

    print("Chargement dataset ok")

    #vérif 
    # x, y = next(iter(train_loader))
    # print(x.shape, y)


    # training loop
    model.train()
    for epoch in range(3):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            print(torch.argmax(logits, dim=1).bincount(minlength=3))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch} | loss={loss.item():.4f}")

    torch.save(model.state_dict(), "c3d_sbd.pth")
    print("✔ Modèle sauvegardé")

if __name__ == "__main__":
    multiprocessing.freeze_support()  # obligatoire sur Windows
    train()


