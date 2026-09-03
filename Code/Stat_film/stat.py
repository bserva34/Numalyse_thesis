from pathlib import Path
import re
import pandas as pd

# Dossier contenant les .txt
DOSSIER = Path("annotations_ms")

films = []

for fichier in DOSSIER.glob("*.txt"):
    # Récupère la décennie depuis le nom (ex: 1970-3 --- Alien.txt)
    m = re.match(r"(\d{4})-", fichier.stem)
    if not m:
        continue

    decennie = int(m.group(1))

    durees = []

    with open(fichier, "r", encoding="utf-8") as f:
        for ligne in f:
            if not ligne.strip():
                continue

            debut, fin = map(float, ligne.split()[:2])
            durees.append(fin - debut)

    if not durees:
        continue

    films.append({
        "film": fichier.stem,
        "decennie": decennie,
        "nb_plans": len(durees),
        "duree_moy_plan_ms": sum(durees) / len(durees),
        "duree_film_ms": sum(durees)
    })

df = pd.DataFrame(films)

# Conversion en secondes
df["duree_moy_plan_s"] = df["duree_moy_plan_ms"] / 1000
df["duree_film_min"] = df["duree_film_ms"] / 60000

print("\n=== Ensemble du corpus ===")
print(f"Nombre de films : {len(df)}")
print(f"Nombre moyen de plans : {df.nb_plans.mean():.1f}")
print(f"Durée moyenne d'un plan : {df.duree_moy_plan_s.mean():.2f} secondes")
print(f"Durée moyenne d'un film : {df.duree_film_min.mean():.2f} minutes")
print(f"Temps total films : {sum(df.duree_film_min):.2f} minutes soit {(sum(df.duree_film_min)/60):.2f} heures soit {((sum(df.duree_film_min)/60)/24):.2f} jours")

par_decennie = (
    df.groupby("decennie")
      .agg(
          films=("film", "count"),
          plans_moy=("nb_plans", "mean"),
          duree_plan_moy_s=("duree_moy_plan_s", "mean"),
          duree_film_moy_min=("duree_film_min", "mean"),
          temps_total=("duree_film_min", "sum")
      )
      .round(2)
)

print("\n=== Par décennie ===")
print(par_decennie)

# Sauvegarde
df.to_csv("stats_par_film.csv", index=False)
par_decennie.to_csv("stats_par_decennie.csv")


import matplotlib.pyplot as plt

# =========================
# GRAPHIQUES PAR DÉCENNIE
# =========================

# On récupère les décennies comme valeurs pour l'axe X
decennies = par_decennie.index.astype(str)

# -------------------------
# GRAPHIQUE 1
# Nombre moyen de plans + durée moyenne d'un plan
# -------------------------

fig, ax1 = plt.subplots(figsize=(10, 6))

# Barres : nombre moyen de plans
bars = ax1.bar(
    decennies,
    par_decennie["plans_moy"],
    alpha=0.7,
    label="Nombre moyen de plans"
)

ax1.set_xlabel("Décennie")
ax1.set_ylabel("Nombre moyen de plans")
ax1.tick_params(axis="y")

# Courbe : durée moyenne d'un plan
ax2 = ax1.twinx()

ax2.plot(
    decennies,
    par_decennie["duree_plan_moy_s"],
    marker="o",
    linewidth=2,
    label="Durée moyenne d'un plan"
)

ax2.set_ylabel("Durée moyenne d'un plan (secondes)")
ax2.tick_params(axis="y")

# Titre
plt.title("Nombre moyen de plans et durée moyenne des plans par décennie")

# Légendes combinées
lignes1, labels1 = ax1.get_legend_handles_labels()
lignes2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(
    lignes1 + lignes2,
    labels1 + labels2,
    loc="upper center"
)

plt.tight_layout()
plt.savefig("fig_stat_plan.png", dpi=300)
# plt.show()



# -------------------------
# GRAPHIQUE 2
# Durée moyenne d'un film + temps total du corpus
# -------------------------

fig, ax = plt.subplots(figsize=(10, 6))

ax.bar(
    decennies,
    par_decennie["duree_film_moy_min"],
    alpha=0.7
)

ax.set_xlabel("Décennie")
ax.set_ylabel("Durée moyenne d'un film (minutes)")
ax.set_title("Durée moyenne des films par décennie")

plt.tight_layout()
plt.savefig("fig_duree_moy.png", dpi=300)
#plt.show()

