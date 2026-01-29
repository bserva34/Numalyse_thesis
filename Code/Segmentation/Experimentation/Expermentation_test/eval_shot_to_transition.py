import json
import os

# --- Chemins à ajuster si nécessaire ---
PATH_JSON = "../../../Dataset/Dataset_Shot/ClipShots/annotations/train+test.json"
PATH_PRED = "AdaptiveDetector_ClipShots"   # dossier contenant les .txt
# ---------------------------------------

# Chargement du JSON
with open(PATH_JSON, "r") as f:
    gt_data = json.load(f)

TP = 0
FP = 0
FN = 0

for video_name, content in gt_data.items():

    gt_transitions = [tuple(t) for t in content["transitions"]]

    pred_file = os.path.join(PATH_PRED, video_name.replace(".mp4", ".txt"))

    predicted_transitions = []

    if os.path.exists(pred_file):

        # Lire les lignes du fichier
        lines = []
        with open(pred_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    lines.append((int(parts[0]), int(parts[1])))

        # Construire TOUTES les paires :
        # (second_element_ligne_i, first_element_ligne_i+1)
        if len(lines) >= 2:
            for i in range(len(lines) - 1):
                second_i = lines[i][1]
                first_next = lines[i + 1][0]
                predicted_transitions.append((second_i-1, first_next))

    else:
        print(f"⚠️ Pas de fichier prédiction pour {video_name}")

    # Comparaison
    set_gt = set(gt_transitions)
    set_pred = set(predicted_transitions)

    TP_local = len(set_gt & set_pred)
    FP_local = len(set_pred - set_gt)
    FN_local = len(set_gt - set_pred)

    TP += TP_local
    FP += FP_local
    FN += FN_local

    #print(f"Vidéo : {video_name} | TP={TP_local}, FP={FP_local}, FN={FN_local}")

# ===========================
# CALCUL DES METRIQUES GLOBALES
# ===========================

Prec = TP / (TP + FP) if (TP + FP) > 0 else 0
Rappel = TP / (TP + FN) if (TP + FN) > 0 else 0
F1 = (2 * Prec * Rappel / (Prec + Rappel)) if (Prec + Rappel) > 0 else 0

# ===========================
# SAUVEGARDE DANS UN FICHIER
# ===========================

video_basename = os.path.splitext(os.path.basename(PATH_JSON))[0]
output_file = PATH_PRED + "_" + video_basename + ".txt"

with open(output_file, "w") as f:
    f.write(f"TP total = {TP}\n")
    f.write(f"FP total = {FP}\n")
    f.write(f"FN total = {FN}\n")
    f.write(f"Précision = {Prec:.4f}\n")
    f.write(f"Rappel = {Rappel:.4f}\n")
    f.write(f"F1-Score = {F1:.4f}\n")

print("\nRésultats sauvegardés dans :", output_file)
