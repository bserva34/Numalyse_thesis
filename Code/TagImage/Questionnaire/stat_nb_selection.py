import pandas as pd

df = pd.read_csv("expert_votes.csv")

methods = [
    "Our_method",
    "Pei",
    "Middle_frame",
    "Zhang_et_al",
    "Chen_et_al"
]

# Nombre de méthodes cochées sur chaque plan
df["n_selected"] = df[methods].sum(axis=1)

stats = (
    df.groupby("Expert")["n_selected"]
      .agg(
          total_selections="sum",
          mean_per_plan="mean",
          std_per_plan="std",
          min_per_plan="min",
          max_per_plan="max"
      )
      .sort_values("total_selections", ascending=False)
)

print(stats.round(2))