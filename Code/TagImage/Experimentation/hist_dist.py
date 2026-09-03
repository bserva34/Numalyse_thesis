import cv2
import os
import argparse
import re

def bhattacharyya_distance(path1, path2):
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)

    if img1 is None or img2 is None:
        raise ValueError(f"Erreur de chargement : {path1} ou {path2}")

    # Option recommandé : HSV
    # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

    h1 = cv2.calcHist([img1], [0, 1, 2], None, [8, 8, 8],
                      [0, 256, 0, 256, 0, 256])
    h2 = cv2.calcHist([img2], [0, 1, 2], None, [8, 8, 8],
                      [0, 256, 0, 256, 0, 256])

    cv2.normalize(h1, h1)
    cv2.normalize(h2, h2)

    return cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA)


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

    output_dir="ablation_studies/"
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
                dist = bhattacharyya_distance(gt_path, method_path)
                line = f"{prefix} {dist:.6f}\n"
                f.write(line)
                #print(line.strip())
            else:
                print(f"[WARNING] Pas d'image méthode pour : {prefix}")
                print(gt_path)

    print(f"\nRésultats écrits dans : {output_name}")
