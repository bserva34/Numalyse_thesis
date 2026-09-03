# version par plan

import pandas as pd
import numpy as np
from itertools import combinations

df = pd.read_csv("expert_votes.csv")

methods = [
    "Our_method",
    "Pei",
    "Middle_frame",
    "Zhang_et_al",
    "Chen_et_al"
]

experts = sorted(df["Expert"].unique())
plans = sorted(df["Plan"].unique())


def masi_similarity(A, B):

    A = set(A)
    B = set(B)

    if len(A) == 0 and len(B) == 0:
        return 1.0

    intersection = len(A & B)
    union = len(A | B)

    if union == 0:
        return 1.0

    jaccard = intersection / union

    if A == B:
        m = 1.0

    elif A.issubset(B) or B.issubset(A):
        m = 2/3

    elif intersection > 0:
        m = 1/3

    else:
        m = 0.0

    return jaccard * m


# ensembles sélectionnés par expert et plan
votes = {}

for plan in plans:

    votes[plan] = {}

    subset = df[df["Plan"] == plan]

    for _, row in subset.iterrows():

        selected = {
            method
            for method in methods
            if row[method] == 1
        }

        votes[plan][row["Expert"]] = selected


# matrice expert-expert
masi_matrix = pd.DataFrame(
    index=experts,
    columns=experts,
    dtype=float
)

for e1 in experts:

    for e2 in experts:

        scores = []

        for plan in plans:

            scores.append(
                masi_similarity(
                    votes[plan][e1],
                    votes[plan][e2]
                )
            )

        masi_matrix.loc[e1, e2] = np.mean(scores)


print("\nMASI similarity matrix\n")
print(masi_matrix.round(3))


# moyenne globale
values = []

for i in range(len(experts)):
    for j in range(i + 1, len(experts)):
        values.append(
            masi_matrix.iloc[i, j]
        )

print(
    f"\nMean MASI similarity = {np.mean(values):.3f}"
)


# expert moyen
mean_similarity = (
    masi_matrix.sum(axis=1) - 1
) / (len(experts) - 1)

print("\nMean MASI per expert\n")
print(
    mean_similarity.sort_values(
        ascending=False
    )
)