import pandas as pd
import numpy as np

df = pd.read_csv("expert_votes.csv")

methods = [
    "Our_method",
    "Pei",
    "Middle_frame",
    "Zhang_et_al",
    "Chen_et_al"
]

experts = sorted(df["Expert"].unique())

profiles = {}

for expert in experts:

    subset = df[df["Expert"] == expert]

    selected = set()

    for _, row in subset.iterrows():

        plan = row["Plan"]

        for method in methods:

            if row[method] == 1:
                selected.add((plan, method))

    profiles[expert] = selected

method_profiles = {}

for method in methods:

    selected = set()

    for _, row in df.iterrows():

        if row[method] == 1:

            selected.add(
                (row["Plan"], row["Expert"])
            )

    method_profiles[method] = selected


def dice(A, B):

    denom = len(A) + len(B)

    if denom == 0:
        return 1.0

    return 2 * len(A & B) / denom


dice_matrix = pd.DataFrame(
    index=experts,
    columns=experts,
    dtype=float
)

for e1 in experts:
    for e2 in experts:
        dice_matrix.loc[e1, e2] = dice(
            profiles[e1],
            profiles[e2]
        )

#print(dice_matrix.round(3))


# Moyenne hors diagonale

values = []

for i in range(len(experts)):
    for j in range(i + 1, len(experts)):
        values.append(
            dice_matrix.iloc[i, j]
        )

# dice_matrix = pd.DataFrame(
#     index=methods,
#     columns=methods,
#     dtype=float
# )

# for e1 in methods:
#     for e2 in methods:
#         dice_matrix.loc[e1, e2] = dice(
#             method_profiles[e1],
#             method_profiles[e2]
#         )

# #print(dice_matrix.round(3))


# # Moyenne hors diagonale

# values = []

# for i in range(len(methods)):
#     for j in range(i + 1, len(methods)):
#         values.append(
#             dice_matrix.iloc[i, j]
#         )

print(
    f"Mean Dice similarity = {np.mean(values):.3f}"
)


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 8))

sns.heatmap(
    dice_matrix.astype(float),
    cmap="viridis",
    vmin=0,
    vmax=1,
    square=True
)

plt.title("Dice similarity between experts")

plt.tight_layout()
plt.show()

mean_similarity = (
    dice_matrix.sum(axis=1) - 1
) / (len(experts) - 1)

print(
    mean_similarity.sort_values(
        ascending=False
    )
)

#version par plan

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# # ==========================
# # Chargement des données
# # ==========================

# df = pd.read_csv("expert_votes.csv")

# methods = [
#     "Our_method",
#     "Pei",
#     "Middle_frame",
#     "Zhang_et_al",
#     "Chen_et_al"
# ]

# experts = sorted(df["Expert"].unique())
# plans = sorted(df["Plan"].unique())

# # ==========================
# # Construction des votes
# # ==========================

# votes = {}

# for plan in plans:

#     votes[plan] = {}

#     subset = df[df["Plan"] == plan]

#     for _, row in subset.iterrows():

#         selected = {
#             method
#             for method in methods
#             if row[method] == 1
#         }

#         votes[plan][row["Expert"]] = selected


# # ==========================
# # Dice coefficient
# # ==========================

# def dice(A, B):

#     if len(A) == 0 and len(B) == 0:
#         return 1.0

#     denom = len(A) + len(B)

#     if denom == 0:
#         return 1.0

#     return 2 * len(A & B) / denom


# # ==========================
# # Matrice Dice
# # ==========================

# dice_matrix = pd.DataFrame(
#     index=experts,
#     columns=experts,
#     dtype=float
# )

# for e1 in experts:

#     for e2 in experts:

#         scores = []

#         for plan in plans:

#             scores.append(
#                 dice(
#                     votes[plan][e1],
#                     votes[plan][e2]
#                 )
#             )

#         dice_matrix.loc[e1, e2] = np.mean(scores)


# print("\nDice matrix\n")
# print(dice_matrix.round(3))

# # ==========================
# # Dice moyen global
# # ==========================

# values = []

# for i in range(len(experts)):
#     for j in range(i + 1, len(experts)):
#         values.append(
#             dice_matrix.iloc[i, j]
#         )

# print(
#     f"\nMean Dice similarity = {np.mean(values):.3f}"
# )

# # ==========================
# # Similarité moyenne expert
# # ==========================

# mean_similarity = (
#     dice_matrix.sum(axis=1) - 1
# ) / (len(experts) - 1)

# print("\nMean similarity per expert\n")
# print(
#     mean_similarity.sort_values(
#         ascending=False
#     )
# )

# # ==========================
# # Heatmap
# # ==========================

# plt.figure(figsize=(10, 8))

# sns.heatmap(
#     dice_matrix.astype(float),
#     cmap="viridis",
#     vmin=0,
#     vmax=1,
#     square=True,
#     annot=True,
#     fmt=".2f"
# )

# plt.title(
#     "Mean plan-wise Dice similarity between experts"
# )

# plt.tight_layout()
# plt.show()