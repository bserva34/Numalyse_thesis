import pandas as pd
import krippendorff

df = pd.read_csv("expert_votes.csv")

methods = [
    "Our_method",
    "Pei",
    "Middle_frame",
    "Zhang_et_al",
    "Chen_et_al"
]

for method in methods:

    # matrice experts × plans
    matrix = df.pivot(
        index="Expert",
        columns="Plan",
        values=method
    )

    alpha = krippendorff.alpha(
        reliability_data=matrix.values,
        level_of_measurement='nominal'
    )

    print(f"{method}: alpha = {alpha:.3f}")