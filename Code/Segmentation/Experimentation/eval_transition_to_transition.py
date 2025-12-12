import json
import os

# --- Chemins à ajuster si besoin ---
PATH_JSON = "../../../Dataset/Dataset_Shot/ClipShots/annotations/train+test.json"
#PATH_JSON = "../../../Dataset/Dataset_Shot/BBC/bbc.json"
PATH_PRED = "AutoShot_ClipShots"      # dossier contenant les fichiers texte
# -----------------------------------

# Chargement du fichier JSON vérité terrain
with open(PATH_JSON, "r") as f:
    gt_data = json.load(f)

TP = 0
FP = 0
FN = 0

for video_name, content in gt_data.items():

    # Récupération des transitions ground truth
    gt_transitions = [tuple(t) for t in content["transitions"]]

    # Fichier texte associé
    pred_file = os.path.join(PATH_PRED, video_name.replace(".mp4", ".txt"))

    predicted_transitions = []

    # Lecture des prédictions (si ficher existe)
    if os.path.exists(pred_file):
        with open(pred_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    predicted_transitions.append((int(parts[0]), int(parts[1])))
    else:
        print(f"⚠️ Fichier prédiction manquant pour : {video_name}")
    
    # --- Comparaison ---
    # On transforme en sets pour faciliter le matching exact
    set_gt = set(gt_transitions)
    set_pred = set(predicted_transitions)

    # Vrai Positifs : intersection exacte
    TP_local = len(set_gt & set_pred)

    # Faux Positifs : prédits mais pas en GT
    FP_local = len(set_pred - set_gt)

    # Faux Négatifs : en GT mais pas prédits
    FN_local = len(set_gt - set_pred)

    # Mise à jour des compteurs globaux
    TP += TP_local
    FP += FP_local
    FN += FN_local

    #print(f"Vidéo: {video_name} | TP={TP_local}, FP={FP_local}, FN={FN_local}")

Prec = TP / (TP+FP)
Rappel = TP / (TP+FN)
F1 = 2 * ((Prec*Rappel)/(Prec+Rappel))

video_basename = os.path.splitext(os.path.basename(PATH_JSON))[0]
output_file = PATH_PRED + "_" + video_basename + ".txt"

with open(output_file, "w") as f:            
    f.write(f"TP total = {TP}\n")
    f.write(f"FP total = {FP}\n")
    f.write(f"FN total = {FN}\n")
    f.write(f"Précision = {Prec}\n")
    f.write(f"Rappel = {Rappel}\n") 
    f.write(f"F1-Score = {F1}\n")

print(f"Résultats enregistrés dans {output_file}")    
