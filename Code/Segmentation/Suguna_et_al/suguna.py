"""
symmetry_article_pipeline.py
Implémentation fidèle à l'article :
'The Detection of Video Shot Transitions Based on Primary Segments ... SURF' (Symmetry 2022)

Modifications ajoutées :
 - fusion des primary segments consécutifs de petite taille (par défaut taille 1).
 - fusion des candidate segments consécutifs de petite taille (par défaut taille 1).
 - suppression des résultats 'none' (considérés non-transitions).
 - paramètre min_segment_size pour contrôler le seuil de fusion.

Usage:
    from symmetry_article_pipeline import detect_transitions_article
    results = detect_transitions_article('video.mp4', debug_plot=True)

Dépendances:
    pip install opencv-python opencv-contrib-python numpy scipy tqdm matplotlib
"""

import cv2
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# -----------------------------
# Utilities / histogram distance
# -----------------------------

def frame_to_hsv_hist(frame, bins=(16,)):
    """Retourne un vecteur histogramme 1D concaténé H, S, V (bins par canal)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    channels = []
    for ch in range(3):
        hist = cv2.calcHist([hsv], [ch], None, [bins[0]], [0, 256]).flatten()
        # normaliser l'histogramme (article parle de fréquences)
        if hist.sum() > 0:
            hist = hist / hist.sum()
        channels.append(hist)
    return np.concatenate(channels)

def chi2_distance(h1, h2, eps=1e-8):
    """Distance chi-square standard (utilisant la formule de comparaison d'histogrammes)."""
    # On utilise la version stable : sum( (h1 - h2)^2 / (h1 + eps) )
    denom = h1 + h2 + eps
    return float(np.sum(((h1 - h2) ** 2) / denom))

def compute_all_diffs(frames, bins=16):
    """Calcule diffs[i] = distance entre frame i et i+1 pour i in [0..N-2]."""
    H = [frame_to_hsv_hist(f, bins=(bins,)) for f in frames]
    diffs = []
    for i in range(len(H) - 1):
        d = chi2_distance(H[i], H[i+1])
        diffs.append(d)
    return np.array(diffs, dtype=float)

# -----------------------------
# Fusion utilities
# -----------------------------
def merge_small_segments(segments, min_size=2):
    """
    Fusionne segments consécutifs de petite taille.
    segments : liste de tuples (s, e) (indices diffs).
    min_size : taille minimale (en nombre d'indices diffs) pour qu'un segment soit considéré 'assez grand'.
               Par défaut 2 => fusionne uniquement les segments de longueur 1.
    Retourne une nouvelle liste de segments fusionnés.
    """
    if not segments:
        return []

    merged = []
    cur_s, cur_e = segments[0]

    for s, e in segments[1:]:
        # si consécutifs (cur_e == s) et le segment courant est trop petit -> fusionner
        if cur_e == s and (cur_e - cur_s) < min_size:
            cur_e = e
        # si le prochain est trop petit et adjacent -> fusionner
        elif (e - s) < min_size and cur_e == s:
            cur_e = e
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e

    merged.append((cur_s, cur_e))
    return merged

def merge_small_candidates_per_primary(candidates, primaries, min_size=2):
    """
    Fusionne les candidates (s,e,forced) mais uniquement À L'INTÉRIEUR
    de chaque segment primaire auquel ils appartiennent.
    Fusion condition : deux candidates adjacentes sont fusionnées si
    elles sont consécutives (cur_e == s) ET (taille_cur < min_size OR taille_next < min_size).
    """
    # 1) construire une liste vide pour chaque primary
    grouped = {i: [] for i in range(len(primaries))}

    # 2) assigner chaque candidate au bon primary (ps <= s < pe)
    for (s, e, f) in candidates:
        assigned = False
        for i, (ps, pe) in enumerate(primaries):
            if ps <= s < pe:
                grouped[i].append((s, e, f))
                assigned = True
                break
        if not assigned:
            # Si candidate en dehors de primaries, on la met à la fin (sécurité)
            grouped.setdefault(-1, []).append((s, e, f))

    # 3) fusion dans chaque primary séparément
    merged_all = []
    for i in sorted(grouped.keys()):
        cand_list = grouped[i]
        if not cand_list:
            continue

        cand_list.sort(key=lambda x: x[0])  # trier par start

        cur_s, cur_e, cur_f = cand_list[0]
        merged = []

        for (s, e, f) in cand_list[1:]:
            # tailles
            cur_size = cur_e - cur_s
            next_size = e - s

            if cur_e == s and (cur_size < min_size or next_size < min_size):
                # fusion : étendre l'intervalle et cumuler forced par AND
                cur_e = e
                cur_f = bool(cur_f and f)
            else:
                merged.append((cur_s, cur_e, cur_f))
                cur_s, cur_e, cur_f = s, e, f

        merged.append((cur_s, cur_e, cur_f))
        merged_all.extend(merged)

    # 4) re-trier globalement par start et retourner
    merged_all.sort(key=lambda x: x[0])
    return merged_all



# -----------------------------
# Primary segments (conforme Algo 1)
# -----------------------------

def primary_segments_from_global_article(diffs, alpha=0.5):
    """
    Version corrigée selon ta nouvelle règle :

    - S'il n'y a qu'un seul boundary : créer 2 segments : (0, b) et (b, len(diffs)).
    - S'il y a plusieurs boundaries : créer des segments uniquement ENTRE boundaries.
      (pas de segment initial ni final)
    """
    if len(diffs) == 0:
        return [], 0.0

    # seuil global
    mu = float(np.mean(diffs))
    sigma = float(np.std(diffs))
    T = mu + alpha * sigma

    # boundaries (indices dans diffs)
    boundary_idxs = np.where(diffs > T)[0].tolist()

    # aucun boundary → aucun primary
    if len(boundary_idxs) == 0:
        return [], float(T)

    # --------
    # CASE 1 : 1 boundary → 2 primary segments couvrant tout
    # --------
    if len(boundary_idxs) == 1:
        b = boundary_idxs[0]
        primaries = [ (b, len(diffs))]
        return primaries, float(T)

    # --------
    # CASE 2 : plusieurs boundaries → segments entre boundaries uniquement
    # --------
    primaries = []
    
    # 1. Segment initial (du début à la première limite)
    # if boundary_idxs[0] > 0:
    #     primaries.append((0, boundary_idxs[0]))
    
    # 2. Segments entre les limites
    for i in range(len(boundary_idxs) - 1):
        s = boundary_idxs[i]
        e = boundary_idxs[i + 1]
        if e > s: # s'assurer que la longueur > 0
            primaries.append((s, e))

    # 3. Segment final (de la dernière limite à la fin)
    last_b = boundary_idxs[-1]
    if last_b < len(diffs):
        primaries.append((last_b, len(diffs)))
    
    # Tri et nettoyage pour le cas d'un seul boundary (qui est maintenant géré par cette logique)
    primaries = sorted(list(set(primaries)), key=lambda x: x[0])

    return primaries, float(T)



# -----------------------------
# Candidate segments (conforme)
# -----------------------------

# def candidate_segments_within_primary_article(diffs, primaries, alpha=0.5):
#     """
#     Pour chaque primary (s,e) (indices diffs), on calcule le seuil local mu+alpha*sigma.
#     POUR CHAQUE i in [s, e): si diffs[i] > local_thresh -> candidate (i, i+1).
#     Si aucun candidate trouvé dans le primary -> forced cut: (e-1, e, forced=True)
#     Référence : section 3.2 + Algorithm 1.
#     """
#     candidates = []
#     for (s, e) in primaries:
#         if e <= s:
#             continue
#         sub = diffs[s:e]
#         mu = float(np.mean(sub)) if len(sub) > 0 else 0.0
#         sigma = float(np.std(sub)) if len(sub) > 0 else 0.0
#         local_th = mu + alpha * sigma
#         found = False
#         for i in range(s, e):
#             if diffs[i] > local_th:
#                 # candidate boundaries are diffs indices i (start) and i+1 (end)
#                 candidates.append((i, i+1, False))
#                 found = True
#         if not found:
#             # forced cut at the last frame in primary segment (article: last frame -> hard cut)
#             last = max(s, e-1)
#             # ensure we don't go out of bounds (end is at most len(diffs))
#             end_idx = last+1 if last+1 <= len(diffs) else last
#             candidates.append((last, end_idx, True))
#     return candidates

def candidate_segments_within_primary_article_max(diffs, primaries, alpha=0.5):
    """
    Variante stricte : un SEUL candidate par segment primaire.
    On choisit le pic le plus élevé (max diff) dépassant le seuil local.
    Si aucun pic ne dépasse le seuil -> forced cut (comme dans l'article).
    """
    candidates = []

    for (s, e) in primaries:
        if e <= s:
            continue

        sub = diffs[s:e]
        if len(sub) == 0:
            continue

        mu = float(np.mean(sub))
        sigma = float(np.std(sub))
        local_th = mu + alpha * sigma

        # indices des valeurs > local threshold
        peak_indices = [i for i in range(s, e) if diffs[i] > local_th]

        if len(peak_indices) == 0:
            # forced cut at last frame of segment
            last = max(s, e-1)
            end_idx = min(last+1, len(diffs))
            candidates.append((last, end_idx, True))
        else:
            # choisir le pic le plus haut
            best_i = max(peak_indices, key=lambda i: diffs[i])
            candidates.append((best_i, best_i+1, False))

    return candidates




# -----------------------------
# Feature detector / matcher (SURF/SIFT/ORB)
# -----------------------------

def get_detector_matcher_article():
    """
    Utilise SURF si disponible, sinon SIFT, sinon ORB.
    Matcher adapté (L2 pour float, Hamming pour binaire).
    """
    detector = None
    is_binary = False
    # SURF (xfeatures2d)
    try:
        if hasattr(cv2, 'xfeatures2d'):
            detector = cv2.xfeatures2d.SURF_create(400)
            is_binary = False
    except Exception:
        detector = None
    if detector is None:
        try:
            detector = cv2.SIFT_create()
            is_binary = False
        except Exception:
            detector = None
    if detector is None:
        detector = cv2.ORB_create(nfeatures=1500)
        is_binary = True

    if is_binary:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    else:
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    return detector, matcher, is_binary

def surf_match_rate(frA, frB, detector=None, matcher=None):
    """
    Calcule un taux de similarité (matching rate) entre deux frames (Fi, Fi+1) :
    num_good_matches / min(num_kpA, num_kpB)
    Filtre adaptatif sur la distance (median*1.5).
    """
    if detector is None or matcher is None:
        detector, matcher, _ = get_detector_matcher_article()

    grayA = cv2.cvtColor(frA, cv2.COLOR_BGR2GRAY) if len(frA.shape) == 3 else frA
    grayB = cv2.cvtColor(frB, cv2.COLOR_BGR2GRAY) if len(frB.shape) == 3 else frB

    kp1, des1 = detector.detectAndCompute(grayA, None)
    kp2, des2 = detector.detectAndCompute(grayB, None)
    n1 = 0 if des1 is None else des1.shape[0]
    n2 = 0 if des2 is None else des2.shape[0]
    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
        return 0.0, 0, n1, n2

    matches = matcher.match(des1, des2)
    if len(matches) == 0:
        return 0.0, 0, n1, n2
    dists = np.array([m.distance for m in matches])
    dth = np.median(dists) * 1.5
    good = [m for m in matches if m.distance <= dth]
    num_good = len(good)
    denom = min(n1, n2) if min(n1, n2) > 0 else 1
    match_rate = num_good / denom
    return float(match_rate), int(num_good), int(n1), int(n2)

# -----------------------------
# Analyze candidate (conforme à article)
# -----------------------------
def analyze_candidate_article_light(video_path,candidate, length, diffs, detector=None, matcher=None, match_threshold=0.5, alpha=0.5):
    s_diff, e_diff, forced = candidate

    # map to frames: Fi = s_diff, Fj = e_diff
    Fs = int(max(0, min(s_diff, length-1)))
    Fe = int(max(0, min(e_diff, length-1)))

    return {
        'type': 'cut',
        'start_frame': Fs,
        'end_frame': Fe,
        'match_rate': 0.0,
        'num_matches': 0,
        'keypoints_A': 0,
        'keypoints_B': 0,
        'num_peaks': 0,
        'local_thresh': None,
        'forced': True
    }


def analyze_candidate_article(video_path,candidate, length, diffs, detector=None, matcher=None, match_threshold=0.5, alpha=0.5):
    """
    Logique conforme aux CASE 1/CASE 2 de l'article (section 3.3).
    candidate : (s_diff, e_diff, forced_flag) où s_diff,e_diff sont indices diffs
    Mapping frames : Fi = s_diff, Fi+1 = e_diff (convention article).
    Référence : section 3.3, Algorithm 1.
    """
    s_diff, e_diff, forced = candidate

    # map to frames: Fi = s_diff, Fj = e_diff
    Fs = int(max(0, min(s_diff, length-1)))
    Fe = int(max(0, min(e_diff, length-1)))

    if forced:
        return {
            'type': 'cut',
            'start_frame': Fs,
            'end_frame': Fe,
            'match_rate': 0.0,
            'num_matches': 0,
            'keypoints_A': 0,
            'keypoints_B': 0,
            'num_peaks': 0,
            'local_thresh': None,
            'forced': True
        }

    cap = cv2.VideoCapture(video_path)

    cap.set(cv2.CAP_PROP_POS_FRAMES, Fs)
    retA, frA = cap.read()

    cap.set(cv2.CAP_PROP_POS_FRAMES, Fe)
    retB, frB = cap.read()

    cap.release()

    if not retA or not retB:
        return None

    # SURF/SIFT/ORB matching between Fi and Fj
    mr, nm, nA, nB = surf_match_rate(frA, frB, detector=detector, matcher=matcher)

    # fenêtre de diffs à l'intérieur (indices diffs entre Fs .. Fe-1 inclus)
    if Fe > Fs:
        window = diffs[Fs:Fe]
    else:
        window = np.array([])

    if len(window) > 0:
        mu = float(np.mean(window))
        sigma = float(np.std(window))
        local_th = mu + alpha * sigma
        peaks_mask = window > local_th
        num_peaks = int(np.sum(peaks_mask))
    else:
        local_th = 0.0
        num_peaks = 0

    # print(num_peaks)
    # print(mr)

    # Application de la logique article :
    # Case 1: match < match_threshold
    if mr < match_threshold:
        if num_peaks == 0:
            t = 'cut'
        else:
            # article: si SURF < threshold et il existe des peaks -> considérer gradual (mais article ambigu)
            # Le papier indique "si SURF < 0.5 et il existe des peaks -> alors gradual si plus de 5 peaks"
            # Ici on suit la logique simplifiée : si il y a des peaks on marque 'gradual' si >5 sinon 'none'
            # if num_peaks > 5:
            #     t = 'gradual'
            # else:
            #     t = 'cut'
            t='none'
    # else:
    #     # Case 2: mr >= threshold
    #     if num_peaks > 5:
    #         t = 'gradual'
    #     else:
    #         t = 'cut'
    else: # Case 2: mr >= threshold
        if num_peaks == 0: 
            t = 'cut'
        else:
            t = 'gradual'

    return {
        'type': t,
        'start_frame': Fs,
        'end_frame': Fe,
        'match_rate': float(mr),
        'num_matches': int(nm),
        'keypoints_A': int(nA),
        'keypoints_B': int(nB),
        'num_peaks': int(num_peaks),
        'local_thresh': float(local_th),
        'forced': False
    }

# -----------------------------
# Pipeline principal
# -----------------------------

def merge_close_transitions(results, max_gap=5):
    if not results:
        return []

    # Trier par start_frame au cas où
    results = sorted(results, key=lambda r: r['start_frame'])

    merged = []
    cur = results[0].copy()

    for nxt in results[1:]:
        gap = nxt['start_frame'] - cur['end_frame']

        if gap <= max_gap:
            # Fusion : étendre l’intervalle
            cur['end_frame'] = max(cur['end_frame'], nxt['end_frame'])
            cur['type'] = 'gradual'
            cur['forced'] = cur.get('forced', False) and nxt.get('forced', False)
        else:
            merged.append(cur)
            cur = nxt.copy()

    merged.append(cur)
    return merged


def detect_transitions_article(video_path,
                               bins=16,
                               alpha=0.5,
                               match_threshold=0.5,
                               debug_plot=False,
                               min_segment_size=2):
    """
    Pipeline entièrement conforme (autant que possible) à Algorithm 1 et aux sections 3.1-3.3,
    avec améliorations pratiques :
      - fusion des primaries/candidates de petite taille (min_segment_size)
      - suppression des résultats 'none'
    Renvoie une liste de dictionnaires (transitions).
    """
    # lire frames
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Impossible d'ouvrir la vidéo: {video_path}")
        # frames = []
        # while True:
        #     ret, fr = cap.read()
        #     if not ret:
        #         break
        #     frames.append(fr)
        # cap.release()

        # if len(frames) < 2:
        #     return []

        # # calcul des diffs histogramme (HSV + chi-square)
        # diffs = compute_all_diffs(frames, bins=bins)

        prev_dist = 0
        diffs = []
        length=0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            length+=1

            #frame = cv2.resize(frame, (320, 180))
            height, width, layers = frame.shape
            new_h = int(height / 6)
            new_w = int(width / 6)
            #print(f"{new_h},{new_w}")
            frame = cv2.resize(frame, (new_w, new_h))

            if prev_dist is not None:
                h1=prev_dist
                h2 = frame_to_hsv_hist(frame)
                diffs.append(chi2_distance(h1, h2))
            else :
                h2=frame_to_hsv_hist(frame)

            prev_dist = h2
        cap.release()

    except Exception as e:
        print(f"[ERREUR] {e}")
        return []

    diffs=np.array(diffs, dtype=float)


    # primary segments (global threshold)
    primaries, global_thresh = primary_segments_from_global_article(diffs, alpha=alpha)
    #print(primaries)

    # fusion des primary segments de petite taille (ex: taille 1)
    # primaries = merge_small_segments(primaries, min_size=min_segment_size)
    # print(primaries)

    # candidate segments (local thresholds inside each primary)
    candidates = candidate_segments_within_primary_article_max(diffs, primaries, alpha=alpha)
    #print(candidates)

    # fusion des candidates de petite taille
    # candidates = merge_small_candidates_per_primary(candidates, primaries, min_size=min_segment_size)
    # print(candidates)

    # prepare detector/matcher once
    # detector, matcher, _ = get_detector_matcher_article()

    results = []
    for cand in candidates:
        # info = analyze_candidate_article(video_path,cand, length, diffs, detector=detector, matcher=matcher,
        #                                  match_threshold=match_threshold, alpha=alpha)
        info = analyze_candidate_article_light(video_path,cand, length, diffs,
                                 match_threshold=match_threshold, alpha=alpha)
        results.append(info)

    # supprimer les 'none' (considérés non-transitions)
    results = [r for r in results if r.get('type') != 'none']

    results = merge_close_transitions(results, max_gap=5)


    if debug_plot:
        _debug_plot(diffs, primaries, candidates, results, global_thresh)

    return results

# -----------------------------
# Debug plot
# -----------------------------

def _debug_plot(diffs, primaries, candidates, results, global_thresh):
    plt.figure(figsize=(14,4))
    plt.plot(diffs, label='hist diffs (χ²)')
    plt.axhline(global_thresh, color='k', linestyle='--', label='global_thresh')
    maxy = max(diffs) if len(diffs) > 0 else 1.0
    # cpt=1
    # for (s,e) in primaries:
    #     plt.axvspan(s, e-1, alpha=0.12, color='purple')
    #     plt.text(s, maxy*0.9,cpt , rotation=90, fontsize=8)
    #     cpt+=1
    # for (s,e,forced) in candidates:
    #     c = 'red' if not forced else 'purple'
    #     plt.axvspan(s, e-1, alpha=0.22, color=c)
    for r in results:
        # annoter le type au niveau start_frame
        c = 'red' if r['type']=='cut' else 'green'
        plt.axvspan(r['start_frame'], r['end_frame'], alpha=0.12, color=c)
        plt.text(r['start_frame'], maxy*0.9, r['type'], rotation=90, fontsize=8)
    plt.show()

# -----------------------------
# CLI
# -----------------------------

def process_videos_in_directory(directory, output_dir):
    # Utilisez exist_ok=True pour éviter une exception si le dossier existe déjà
    os.makedirs(output_dir, exist_ok=True)
    
    video_files = [f for f in os.listdir(directory) if f.lower().endswith(('.mp4', '.avi', '.mkv', '.mov', '.m4v', '.mpe'))]

    for video in video_files:
        video_path = os.path.join(directory, video)
        video_basename = os.path.splitext(os.path.basename(video_path))[0]
        output_file = os.path.join(output_dir, f"{video_basename}.txt")
        if os.path.exists(output_file):
            print(f"Fichier de résultats {output_file} existe déjà")
            continue
        res = detect_transitions_article(video_path)
        write_res(output_dir,video_path,res)



def write_res(directory, video_path, res):
    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    output_file = os.path.join(directory, f"{video_basename}.txt")

    with open(output_file, "w") as f:
        for item in res:
            start = item.get('start_frame')
            end = item.get('end_frame')
            f.write(f"{start}\t{end}\n")

    print(f"Résultats enregistrés dans {output_file}")    



if __name__ == '__main__':
    # import argparse
    # parser = argparse.ArgumentParser(description='Pipeline fidèle à l\'article (primary->candidate->SURF) avec améliorations')
    # parser.add_argument('video', help='chemin vers la vidéo')
    # parser.add_argument('--bins', type=int, default=16)
    # parser.add_argument('--alpha', type=float, default=0.5)
    # parser.add_argument('--match_threshold', type=float, default=0.5)
    # parser.add_argument('--min_segment_size', type=int, default=2, help='taille min segments (2 => fusionne segments de taille 1)')
    # parser.add_argument('--debug', action='store_true')
    # args = parser.parse_args()

    # res = detect_transitions_article(args.video, bins=args.bins, alpha=args.alpha,
    #                                  match_threshold=args.match_threshold,
    #                                  debug_plot=args.debug,
    #                                  min_segment_size=args.min_segment_size)
    # for r in res:
    #     print(r)

    # import argparse
    # parser = argparse.ArgumentParser(description='DMD detector')
    # parser.add_argument('list', help='chemin list')
    # args = parser.parse_args()

    # input_directory = r"../../../Dataset/Dataset_Shot/V3C/V3C1/videos" 

    # ech_basename = os.path.splitext(os.path.basename(args.list))[0]

    # output_directory = r"../Experimentation/Suguna_V3C1"  # Dossier de sortie
    # output_directory = output_directory+"_"+ech_basename

    # with open(args.list, "r", encoding="utf-8") as f:
    #     for line in f:
    #         line = line.strip()  # enlève \n et les espaces
    #         #print("Traitement de la vidéo : ",line)
    #         dir_prov=os.path.join(input_directory, line)
    #         process_videos_in_directory(dir_prov, output_directory)
    

    input_directory = r"../../../Dataset/Dataset_Shot/AutoShot/video" 
    output_directory = r"../Experimentation/AutoShot_TEST/Suguna_AutoShot"  # Dossier de sortie
    process_videos_in_directory(input_directory, output_directory)