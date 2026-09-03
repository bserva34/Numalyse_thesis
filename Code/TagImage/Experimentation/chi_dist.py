import cv2
import os
import argparse
import re
import numpy as np

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



def extract_prefix_gt(fname):
    name = os.path.splitext(fname)[0]
    return re.split(r'_\d{2}[-_]\d{2}[-_]\d{2}\[\d{2}\]', name)[0]

def extract_prefix_method(fname):
    name = os.path.splitext(fname)[0]
    return name.split('_keyframe')[0]


def build_prefix_dict(folder, prefix_func):
    d = {}
    for fname in os.listdir(folder):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            prefix = prefix_func(fname)
            d[prefix] = os.path.join(folder, fname)
    return d


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("gt_dir", help="Dossier des valeurs de vérité")
    parser.add_argument("method_dir", help="Dossier des images méthode")
    args = parser.parse_args()

    gt_dir = args.gt_dir
    method_dir = args.method_dir

    gt_images = build_prefix_dict(gt_dir, extract_prefix_gt)
    method_images = build_prefix_dict(method_dir, extract_prefix_method)

    output_dir="Comparaison_methode_chi2/"
    os.makedirs(output_dir, exist_ok=True)

    # Nom du fichier de sortie = nom du dossier méthode
    output_name = os.path.basename(os.path.normpath(method_dir)) + ".txt"
    output_name=output_dir+output_name

    print("Taille GT : ",len(gt_images))
    print("Taille Méthode : ",len(method_images))
    with open(output_name, "w") as f:
        for prefix, gt_path in gt_images.items():
            if prefix in method_images:
                method_path = method_images[prefix]


                dist = chi2_distance(frame_to_hsv_hist(cv2.imread(gt_path)),frame_to_hsv_hist(cv2.imread(method_path)))
                line = f"{prefix} {dist:.6f}\n"
                f.write(line)
                #print(line.strip())
            else:
                print(f"[WARNING] Pas d'image méthode pour : {prefix}")
                print(gt_path)

    print(f"\nRésultats écrits dans : {output_name}")
