import numpy as np
import pandas as pd

# Comptages de sélection
data = {
    "Plan": [f"Plan{i}" for i in range(1, 11)],
    "Our_method":   [10, 5, 6, 5, 14, 15, 9, 3, 2, 13],
    "Pei":          [0, 9, 2, 0, 2, 4, 7, 7, 3, 0],
    "Middle_frame": [9, 4, 2, 6, 7, 2, 9, 11, 0, 11],
    "Zhang_et_al":  [1, 3, 5, 6, 1, 3, 3, 4, 3, 9],
    "Chen_et_al":   [5, 3, 1, 8, 12, 3, 12, 7, 8, 3],
}

df = pd.DataFrame(data)

# Nombre de méthodes
n_methods = 5

def shannon_entropy(counts):
    counts = np.array(counts, dtype=float)

    total = counts.sum()
    probs = counts / total

    # supprimer les zéros pour éviter log(0)
    probs = probs[probs > 0]

    H = -np.sum(probs * np.log(probs))

    H_max = np.log(n_methods)

    H_norm = H / H_max

    return H, H_norm

results = []

for _, row in df.iterrows():

    counts = row[
        ["Our_method",
         "Pei",
         "Middle_frame",
         "Zhang_et_al",
         "Chen_et_al"]
    ].values

    H, H_norm = shannon_entropy(counts)

    results.append({
        "Plan": row["Plan"],
        "Entropy": H,
        "Entropy_normalized": H_norm,
        "Consensus_index": 1 - H_norm
    })

results_df = pd.DataFrame(results)

print(results_df.round(3))


import matplotlib.pyplot as plt

# plt.figure(figsize=(8,4))
# plt.bar(results_df["Plan"], results_df["Consensus_index"])
# plt.ylabel("Consensus index")
# plt.ylim(0,1)
# plt.tight_layout()
# plt.show()