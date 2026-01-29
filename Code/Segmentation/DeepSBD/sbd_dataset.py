# import os
# import numpy as np
# import torch
# from torch.utils.data import Dataset

# LABELS = {
#     "no_transition": 0,
#     "sharp": 1,
#     "gradual": 2
# }

# class SBDDataset(Dataset):
#     def __init__(self, root):
#         self.samples = []

#         for cls in LABELS:
#             cls_path = os.path.join(root, cls)
#             for f in os.listdir(cls_path):
#                 self.samples.append((os.path.join(cls_path, f), LABELS[cls]))

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         path, label = self.samples[idx]
#         frames = np.load(path).astype(np.float32) / 255.0
#         frames = torch.from_numpy(frames).permute(3,0,1,2)
#         return frames, label

import lmdb
import pickle
import torch
from torch.utils.data import Dataset

class LMDBVideoDataset(Dataset):
    def __init__(self, lmdb_path):
        self.lmdb_path = lmdb_path
        self.env = None

        # lecture longueur UNE SEULE FOIS (safe)
        env = lmdb.open(lmdb_path, readonly=True, lock=False)
        with env.begin() as txn:
            length_bytes = txn.get(b"__len__")
            if length_bytes is None:
                raise RuntimeError("LMDB invalide (clé __len__ absente)")
            self.length = pickle.loads(length_bytes)
        env.close()

    def _init_env(self):
        if self.env is None:
            self.env = lmdb.open(
                self.lmdb_path,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False
            )

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        self._init_env()

        key = f"{idx:010d}".encode("ascii")
        with self.env.begin() as txn:
            data = pickle.loads(txn.get(key))

        #x = torch.from_numpy(data["video"])
        x = torch.from_numpy(data["video"]).float() / 255.0

        y = torch.tensor(data["label"], dtype=torch.long)
        return x, y
