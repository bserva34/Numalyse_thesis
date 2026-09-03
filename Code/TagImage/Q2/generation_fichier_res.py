import json
import csv

# Fichiers d'entrée
TXT_FILE = "liste_video.txt"
JSON_FILE = "RES_Brut/BS.json"
CSV_FILE = "BS.csv"

# Colonnes à créer
METHODES = ["Chen", "Faster", "Middle", "Pei", "Yolom", "Zhang"]

# Lecture du fichier JSON
with open(JSON_FILE, "r", encoding="utf-8") as f:
    annotations = json.load(f)

# Dictionnaire : nom_video -> liste des méthodes sélectionnées
selection = {
    item["video"]: set(item["selected_images"])
    for item in annotations
}

# Lecture des vidéos
with open(TXT_FILE, "r", encoding="utf-8") as f:
    videos = [ligne.strip() for ligne in f if ligne.strip()]

# Écriture du CSV
with open(CSV_FILE, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)

    # En-tête
    writer.writerow(["video/plan"] + METHODES)

    # Lignes
    for video in videos:
        choix = selection.get(video, set())
        ligne = [video] + [1 if m in choix else 0 for m in METHODES]
        writer.writerow(ligne)

print(f"CSV créé : {CSV_FILE}")