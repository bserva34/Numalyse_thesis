import os
import numpy as np

# ===================== CONFIG =====================
# PATH_GT = "../../../Dataset/Dataset_Shot/V3C/V3C1/msb"
#PATH_GT = "../../../Dataset/Dataset_Shot/AutoShot/annotations"
#PATH_GT = "../../../Dataset/Dataset_Shot/ClipShots/annotations/annotations"

#PATH_GT = "../../../Dataset/Dataset_Shot/BBC/annotations"
PATH_GT = "../../../Dataset/Dataset_Shot/BBC/annotations"
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

def read_gt_plans_txt(file_path):
    """
    Lecture des plans GT depuis un fichier .txt
    Format attendu :
        start end
        start end
        ...
    """
    plans = []

    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    start = int(parts[0])
                    end = int(parts[1])
                    plans.append((start, end))
                except ValueError:
                    # Ignore les lignes mal formées
                    continue

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


def match_transitions_with_tolerance(gt_transitions, pred_transitions, tol):
    """
    Matching 1-à-1 entre transitions GT et prédites avec tolérance
    """
    gt_used = set()
    TP = 0

    tab_front = []

    for p_end, p_start in pred_transitions:
        #print(f"PREDICTION : {p_end}/{p_start}")

        for idx, (g_end, g_start) in enumerate(gt_transitions):
            if idx in gt_used:
                continue
            #print(f"GT : {g_end}/{g_start} ")

            if abs(p_end - g_end) <= tol and abs(p_start - g_start) <= tol:
                TP += 1
                gt_used.add(idx)

                tab_front.append(abs(p_end - g_end))
                tab_front.append(abs(p_start - g_start))

                #print(f"Prediction :{p_end}/{p_start} - GT : {g_end}/{g_start} ")

                break

    FP = len(pred_transitions) - TP
    FN = len(gt_transitions) - TP

    return TP, FP, FN, tab_front

def match_transitions_with_tolerance_cool(gt_transitions, pred_transitions, tol):
    """
    Matching 1-à-1 entre transitions GT et prédites avec tolérance
    """
    gt_used = set()
    TP = 0

    cpt_tab = np.zeros(len(gt_transitions))

    for p_end, p_start in pred_transitions:
        for idx, (g_end, g_start) in enumerate(gt_transitions):

            if abs(p_end - g_end) <= tol or abs(p_start - g_start) <= tol:
                if idx in gt_used:
                    cpt_tab[idx]+=1
                    continue
                TP += 1
                gt_used.add(idx)
                cpt_tab[idx]+=1
                break

    FP = len(pred_transitions) - TP
    FN = len(gt_transitions) - TP


    return TP, FP, FN, cpt_tab


def match_middle_transitions(gt_transitions, pred_transitions, tol):
    gt_used = set()
    TP = 0
    for p_end, p_start in pred_transitions:

        p_mid = (p_end + p_start) / 2

        for idx, (g_end, g_start) in enumerate(gt_transitions):

            g_mid = (g_end + g_start) / 2

            if abs(p_mid- g_mid) <= tol :
                if idx in gt_used:
                    continue
                TP += 1
                gt_used.add(idx)
                break

    FP = len(pred_transitions) - TP
    FN = len(gt_transitions) - TP

    return TP, FP, FN

# ===================================================


# ===================== MAIN =====================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Calcul performence SBD method')
    parser.add_argument('m', default="Adaptive_V3C1_bis", help='Method path')
    parser.add_argument('t',type=int, default=5, help='TOLERANCE')
    args = parser.parse_args()

    TOLERANCE=args.t
    METHOD_PATH=args.m

    name = os.path.basename(os.path.splitext(METHOD_PATH)[0])
    left_part = os.path.dirname(METHOD_PATH)


    OUTPUT_FILE = left_part+"/Scores/"+ name + f"_results_tol_{TOLERANCE}.txt"

results = []

entries = sorted(os.listdir(METHOD_PATH))

# Si aucun sous-dossier → on considère METHOD_PATH comme un seul sample
if not any(os.path.isdir(os.path.join(METHOD_PATH, e)) for e in entries):
    samples = {"global": METHOD_PATH}
else:
    samples = {
        e: os.path.join(METHOD_PATH, e)
        for e in entries
        if os.path.isdir(os.path.join(METHOD_PATH, e))
    }

for sample_name, sample_path in samples.items():

    TP = FP = FN = 0
    TP_cool = FP_cool = FN_cool = 0
    TP_mid = FP_mid = FN_mid = 0
    video_count = 0
    cpt_tab = []
    tab_front = []


    for pred_filename in os.listdir(sample_path):
        if not pred_filename.endswith(".txt"):
            continue

        video_name = os.path.splitext(pred_filename)[0]

        pred_file = os.path.join(sample_path, pred_filename)
        #gt_file = os.path.join(PATH_GT, video_name + ".tsv")
        gt_file = os.path.join(PATH_GT, video_name + ".txt")

        if not os.path.exists(gt_file):
            print(f"⚠️ GT manquant pour {video_name}")
            continue

        video_count += 1

        # ===== Lecture GT =====
        # gt_plans = read_gt_plans_tsv(gt_file)
        gt_plans = read_gt_plans_txt(gt_file)

        #gt_transitions = read_gt_plans_txt(gt_file) #Other TEST
        gt_transitions = plans_to_transitions(gt_plans) #BBC TEST 

        # ===== Lecture prédictions =====
        pred_transitions = read_pred_transitions_txt(pred_file)

        # ===== Matching =====
        TP_loc, FP_loc, FN_loc, tab_front_prov = match_transitions_with_tolerance(
            gt_transitions, pred_transitions, TOLERANCE
        )

        TP_loc_cool, FP_loc_cool, FN_loc_cool, cpt_tab_prov = match_transitions_with_tolerance_cool(
            gt_transitions, pred_transitions, TOLERANCE
        )

        TP_loc_mid, FP_loc_mid, FN_loc_mid = match_middle_transitions(
            gt_transitions, pred_transitions, TOLERANCE
        )

        TP += TP_loc
        FP += FP_loc
        FN += FN_loc

        TP_cool += TP_loc_cool
        FP_cool += FP_loc_cool
        FN_cool += FN_loc_cool

        TP_mid += TP_loc_mid
        FP_mid += FP_loc_mid
        FN_mid += FN_loc_mid

        cpt_tab=np.concatenate((cpt_tab,cpt_tab_prov))
        tab_front=np.concatenate((tab_front,tab_front_prov))

    Precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    Recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    F1 = 2 * Precision * Recall / (Precision + Recall) if (Precision + Recall) > 0 else 0

    Precision_cool = TP_cool / (TP_cool + FP_cool) if (TP_cool + FP_cool) > 0 else 0
    Recall_cool = TP_cool / (TP_cool + FN_cool) if (TP_cool + FN_cool) > 0 else 0
    F1_cool = 2 * Precision_cool * Recall_cool / (Precision_cool + Recall_cool) if (Precision_cool + Recall_cool) > 0 else 0

    Precision_mid = TP_mid / (TP_mid + FP_mid) if (TP_mid + FP_mid) > 0 else 0
    Recall_mid = TP_mid / (TP_mid + FN_mid) if (TP_mid + FN_mid) > 0 else 0
    F1_mid = 2 * Precision_mid * Recall_mid / (Precision_mid + Recall_mid) if (Precision_mid + Recall_mid) > 0 else 0

    valid_cpt = cpt_tab[cpt_tab != 0]

    if len(valid_cpt) > 0:
        moyenne = np.mean(valid_cpt)
        max_m = np.max(valid_cpt)
    else:
        moyenne = 0
        max_m = 0

    if len(tab_front) > 0:
        moyenne_front = np.mean(tab_front)
    else:
        moyenne_front = 0



    results.append({
        "sample": sample_name,
        "videos": video_count,
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "Precision": Precision,
        "Recall": Recall,
        "F1": F1,
        "TP_cool": TP_cool,
        "FP_cool": FP_cool,
        "FN_cool": FN_cool,
        "Precision_cool": Precision_cool,
        "Recall_cool": Recall_cool,
        "F1_cool": F1_cool,
        "Moyenne": moyenne,
        "Max_m": max_m,
        "Moyenne_front" : moyenne_front,
        "TP_mid": TP_mid,
        "FP_mid": FP_mid,
        "FN_mid": FN_mid,
        "Precision_mid": Precision_mid,
        "Recall_mid": Recall_mid,
        "F1_mid": F1_mid
    })


# # Sauvegarde données brutes pour visualisation ultérieure
# np.save("Scores/"+METHOD_PATH + f"_front_errors_tol_{TOLERANCE}.npy", tab_front)
# np.save("Scores/"+METHOD_PATH + f"_multiple_matches_tol_{TOLERANCE}.npy", cpt_tab)


# ===================== STATS =====================
precisions = np.array([r["Precision"] for r in results])
recalls = np.array([r["Recall"] for r in results])
f1s = np.array([r["F1"] for r in results])

moyennes_front = np.array([r["Moyenne_front"] for r in results])

stats = {
    "Precision": (precisions.mean(), precisions.std(), precisions.min(), precisions.max()),
    "Recall": (recalls.mean(), recalls.std(), recalls.min(), recalls.max()),
    "F1": (f1s.mean(), f1s.std(), f1s.min(), f1s.max()),
    "Frontières" : (moyennes_front.mean(),0,0,0)
}

precisions_cool = np.array([r["Precision_cool"] for r in results])
recalls_cool = np.array([r["Recall_cool"] for r in results])
f1s_cool = np.array([r["F1_cool"] for r in results])

moyennes = np.array([r["Moyenne"] for r in results])
maxs = np.array([r["Max_m"] for r in results])


stats_cool = {
    "Precision_cool": (precisions_cool.mean(), precisions_cool.std(), precisions_cool.min(), precisions_cool.max()),
    "Recall_cool": (recalls_cool.mean(), recalls_cool.std(), recalls_cool.min(), recalls_cool.max()),
    "F1_cool": (f1s_cool.mean(), f1s_cool.std(), f1s_cool.min(), f1s_cool.max()),
    "Moyenne": (moyennes.mean(),0,0,maxs.max())
}
# =================================================


# ===================== SAVE =====================
with open(OUTPUT_FILE, "w") as f:
    f.write(f"=== Évaluation transitions | Tolérance = {TOLERANCE} frames ===\n")
    f.write(f"Méthode : {METHOD_PATH}\n\n")

    f.write(f"STRICT\n")
    for r in results:
        f.write(
            f"{r['sample']} | vidéos={r['videos']} | "
            f"TP={r['TP']} FP={r['FP']} FN={r['FN']} | "
            f"P={r['Precision']:.4f} "
            f"R={r['Recall']:.4f} "
            f"F1={r['F1']:.4f} | "
            f"Frontières={r['Moyenne_front']:.4f}\n"
        )

    f.write("\n=== Statistiques globales (sur échantillons) ===\n")
    for metric, (mean, std, mn, mx) in stats.items():
        f.write(
            f"{metric} | mean={mean:.4f} std={std:.4f} "
            f"min={mn:.4f} max={mx:.4f}\n"
        )

    f.write(f"\n\nCOOL\n")
    for r in results:
        f.write(
            f"{r['sample']} | vidéos={r['videos']} | "
            f"TP={r['TP_cool']} FP={r['FP_cool']} FN={r['FN_cool']} | "
            f"P={r['Precision_cool']:.4f} "
            f"R={r['Recall_cool']:.4f} "
            f"F1={r['F1_cool']:.4f} | "
            f"Moyenne={r['Moyenne']:.4f}\n"
        )

    f.write("\n=== Statistiques globales (sur échantillons) ===\n")
    for metric, (mean, std, mn, mx) in stats_cool.items():
        f.write(
            f"{metric} | mean={mean:.4f} std={std:.4f} "
            f"min={mn:.4f} max={mx:.4f}\n"
        )


    f.write(f"\n\nMID\n")
    for r in results:
        f.write(
            f"{r['sample']} | vidéos={r['videos']} | "
            f"TP={r['TP_mid']} FP={r['FP_mid']} FN={r['FN_mid']} | "
            f"P={r['Precision_mid']:.4f} "
            f"R={r['Recall_mid']:.4f} "
            f"F1={r['F1_mid']:.4f}\n"
        )

        f.write(f"Precision_mid | mean={r['Precision_mid']:.4f}\n")
        f.write(f"Recall_mid | mean={r['Recall_mid']:.4f}\n")
        f.write(f"F1_mid | mean={r['F1_mid']:.4f}")



print(f"✅ Résultats sauvegardés dans {OUTPUT_FILE}")
# =================================================
