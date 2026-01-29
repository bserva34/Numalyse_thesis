# import os

# # --- Chemins ---
# PATH_GT = "../../../Dataset/Dataset_Shot/V3C/V3C1/msb"  # dossier contenant les .tsv (vérité terrain)
# # PATH_PRED = "AutoShot_V3C1_echantillon_1"    # dossier contenant les .txt (prédictions)
# PATH_PRED = "Adaptive_V3C1_echantillon_1"    # dossier contenant les .txt (prédictions)
# # ----------------

# TP = 0
# FP = 0
# FN = 0

# # Parcours de tous les fichiers de prédiction
# for pred_filename in os.listdir(PATH_PRED):

#     if not pred_filename.endswith(".txt"):
#         continue

#     video_name = os.path.splitext(pred_filename)[0]

#     pred_file = os.path.join(PATH_PRED, pred_filename)
#     gt_file = os.path.join(PATH_GT, video_name + ".tsv")

#     # --- Lecture des prédictions ---
#     predicted_transitions = []
#     with open(pred_file, "r") as f:
#         for line in f:
#             parts = line.strip().split()
#             if len(parts) == 2:
#                 predicted_transitions.append((int(parts[0]), int(parts[1])))

#     # --- Lecture GT (plans) ---
#     if not os.path.exists(gt_file):
#         print(f"⚠️ Fichier GT manquant pour {video_name}")
#         continue

#     plans = []
#     with open(gt_file, "r") as f:
#         next(f)  # sauter l'en-tête
#         for line in f:
#             parts = line.strip().split()
#             if len(parts) >= 3:
#                 startframe = int(parts[0])
#                 endframe = int(parts[2])
#                 plans.append((startframe, endframe))

#     # --- Conversion plans → transitions GT ---
#     gt_transitions = []
#     for i in range(len(plans) - 1):
#         end_prev = plans[i][1]
#         start_next = plans[i + 1][0]
#         gt_transitions.append((end_prev, start_next))

#     # --- Comparaison ---
#     set_gt = set(gt_transitions)
#     set_pred = set(predicted_transitions)

#     TP_local = len(set_gt & set_pred)
#     FP_local = len(set_pred - set_gt)
#     FN_local = len(set_gt - set_pred)

#     TP += TP_local
#     FP += FP_local
#     FN += FN_local

#     # Debug optionnel
#     # print(f"{video_name} | TP={TP_local}, FP={FP_local}, FN={FN_local}")

# # --- Métriques globales ---
# Precision = TP / (TP + FP) if (TP + FP) > 0 else 0
# Recall = TP / (TP + FN) if (TP + FN) > 0 else 0
# F1 = 2 * Precision * Recall / (Precision + Recall) if (Precision + Recall) > 0 else 0

# # --- Sauvegarde ---
# output_file = PATH_PRED + "_results.txt"
# with open(output_file, "w") as f:
#     f.write(f"TP total = {TP}\n")
#     f.write(f"FP total = {FP}\n")
#     f.write(f"FN total = {FN}\n")
#     f.write(f"Précision = {Precision}\n")
#     f.write(f"Rappel = {Recall}\n")
#     f.write(f"F1-Score = {F1}\n")

# print(f"Résultats enregistrés dans {output_file}")


import os

# ===================== CONFIG =====================
PATH_GT = "../../../Dataset/Dataset_Shot/V3C/V3C1/msb"
#PATH_PRED = "Adaptive_V3C1_echantillon_1"
PATH_PRED = "AutoShot_V3C1_echantillon_1"

# Choix du mode pour les prédictions :
# "transitions"  -> prédictions = transitions
# "plans"        -> prédictions = plans
PRED_MODE = "transitions"
#PRED_MODE = "plans"
# =================================================


# ===================== METRICS =====================
TP = 0
FP = 0
FN = 0
# =================================================


# ===================== FONCTIONS =====================
def read_gt_plans_tsv(file_path):
    """
    Lecture des plans GT depuis un fichier .tsv (avec header)
    Format attendu : startframe ... endframe
    """
    plans = []
    with open(file_path, "r") as f:
        next(f)  # sauter le header
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                start = int(parts[0])
                end = int(parts[2])
                plans.append((start, end))
    return plans


def read_pred_plans_txt(file_path):
    """
    Lecture des plans prédits depuis un fichier .txt (sans header)
    Format attendu : start end
    """
    plans = []
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                start = int(parts[0])
                end = int(parts[1])
                plans.append((start, end))
    return plans


def read_pred_transitions_txt(file_path):
    """
    Lecture des transitions prédites depuis un fichier .txt
    Format attendu : end_prev start_next
    """
    transitions = []
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                transitions.append((int(parts[0]), int(parts[1])))
    return transitions


def plans_to_transitions(plans):
    """
    Conversion d'une liste de plans en transitions
    """
    transitions = []
    for i in range(len(plans) - 1):
        transitions.append((plans[i][1], plans[i + 1][0]))
    return transitions
# ===================================================


# ===================== MAIN LOOP =====================
for pred_filename in os.listdir(PATH_PRED):

    if not pred_filename.endswith(".txt"):
        continue

    video_name = os.path.splitext(pred_filename)[0]

    pred_file = os.path.join(PATH_PRED, pred_filename)
    gt_file = os.path.join(PATH_GT, video_name + ".tsv")

    if not os.path.exists(gt_file):
        print(f"⚠️ GT manquant pour {video_name}")
        continue

    # ===== GT (toujours des plans) =====
    gt_plans = read_gt_plans_tsv(gt_file)
    gt_transitions = plans_to_transitions(gt_plans)

    # ===== Prédictions =====
    if PRED_MODE == "transitions":
        pred_transitions = read_pred_transitions_txt(pred_file)

    elif PRED_MODE == "plans":
        pred_plans = read_pred_plans_txt(pred_file)
        pred_transitions = plans_to_transitions(pred_plans)

    else:
        raise ValueError("PRED_MODE doit être 'transitions' ou 'plans'")

    # ===== Comparaison =====
    set_gt = set(gt_transitions)
    set_pred = set(pred_transitions)

    TP_local = len(set_gt & set_pred)
    FP_local = len(set_pred - set_gt)
    FN_local = len(set_gt - set_pred)

    TP += TP_local
    FP += FP_local
    FN += FN_local

    # Debug si besoin
    # print(f"{video_name} | TP={TP_local}, FP={FP_local}, FN={FN_local}")
# ===================================================


# ===================== METRICS =====================
Precision = TP / (TP + FP) if (TP + FP) > 0 else 0
Recall = TP / (TP + FN) if (TP + FN) > 0 else 0
F1 = 2 * Precision * Recall / (Precision + Recall) if (Precision + Recall) > 0 else 0
# =================================================


# ===================== SAVE =====================
output_file = PATH_PRED + f"_results.txt"
with open(output_file, "w") as f:
    f.write(f"TP total = {TP}\n")
    f.write(f"FP total = {FP}\n")
    f.write(f"FN total = {FN}\n")
    f.write(f"Précision = {Precision}\n")
    f.write(f"Rappel = {Recall}\n")
    f.write(f"F1-Score = {F1}\n")

print(f"✅ Résultats enregistrés dans {output_file}")
# =================================================
