import cv2
import os
import argparse
import re
import numpy as np

from sklearn.metrics import mutual_info_score
from PIL import Image

def compare_images(path1, path2, bins=64):
    """
    Calcule l'Information Mutuelle (MI) entre deux images.
    Plus le score est élevé, plus les images sont similaires.
    """
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)

    if img1 is None or img2 is None:
        raise ValueError(f"Erreur de chargement : {path1} ou {path2}")

    # On redimensionne si les dimensions ne correspondent pas
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # Conversion en niveaux de gris pour la MI
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Calcul de l'histogramme 2D joint entre les deux images
    # Réduire le nombre de bins (ex: 64 au lieu de 256) améliore la vitesse et réduit le bruit
    hist_2d, _, _ = np.histogram2d(
        gray1.ravel(), 
        gray2.ravel(), 
        bins=bins, 
        range=[[0, 256], [0, 256]]
    )

    # Calcul de la Mutual Information à partir de la distribution jointe
    mi = mutual_info_score(None, None, contingency=hist_2d)

    return mi

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

    output_dir="Comparaison_methode_MI/"
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


                dist = compare_images(gt_path,method_path)
                line = f"{prefix} {dist:.6f}\n"
                f.write(line)
                #print(line.strip())
            else:
                print(f"[WARNING] Pas d'image méthode pour : {prefix}")
                print(gt_path)

    print(f"\nRésultats écrits dans : {output_name}")
