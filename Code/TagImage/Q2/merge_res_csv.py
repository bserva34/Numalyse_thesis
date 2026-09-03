import pandas as pd
from pathlib import Path

# Dossier contenant les CSV des annotateurs
DOSSIER = Path("RES_CSV")

METHODES = ["Chen", "Faster", "Middle", "Pei", "Yolom", "Zhang"]

# ---------- Fusion des fichiers ----------
fichiers = list(DOSSIER.glob("*.csv"))

if not fichiers:
    raise FileNotFoundError("Aucun fichier CSV trouvé.")

dfs = []
for f in fichiers:
    df = pd.read_csv(f)
    df["source"] = f.stem
    dfs.append(df)

resultats = pd.concat(dfs, ignore_index=True)
resultats.to_csv("resultats_combines.csv", index=False, encoding="utf-8-sig")

# ---------- Taux de sélection global ----------
nb_annotations = len(resultats)

stats = []
for m in METHODES:
    stats.append({
        "Méthode": m,
        "Sélections": int(resultats[m].sum()),
        "Total annotations": nb_annotations,
        "Taux sélection (%)": round(resultats[m].mean() * 100, 2)
    })

df_stats = pd.DataFrame(stats)

# ---------- Méthode la plus votée par vidéo ----------
votes = resultats.groupby("video/plan")[METHODES].sum()

victoires = {m: 0 for m in METHODES}

for _, ligne in votes.iterrows():
    max_vote = ligne.max()

    # Ignore les vidéos où personne n'a voté
    if max_vote == 0:
        continue

    gagnants = ligne[ligne == max_vote].index
    for g in gagnants:
        victoires[g] += 1

nb_videos_avec_votes = (votes.max(axis=1) > 0).sum()

df_stats["Victoires"] = df_stats["Méthode"].map(victoires)
df_stats["Taux plus votée (%)"] = (
    df_stats["Victoires"] / nb_videos_avec_votes * 100
).round(2)

# ---------- Sauvegarde ----------
df_stats.to_csv("statistiques_methodes.csv", index=False, encoding="utf-8-sig")

print(df_stats)