import os
import re
import argparse

from PIL import Image

import torch
from transformers import AutoProcessor, AutoModel


# =========================================================
# Chargement modèle DINOv2 (Vision-Vision)
# =========================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

# Utilisation de la version Large pour être équivalent à CLIP ViT-L
model_name = "facebook/dinov2-large"

model = AutoModel.from_pretrained(model_name).to(device)
processor = AutoProcessor.from_pretrained(model_name)

model.eval()


# =========================================================
# Embedding image avec DINOv2
# =========================================================

def get_dinov2_embedding(image_path):

    image = Image.open(image_path).convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        
        # Le pooler_output extrait le descripteur global de l'image (dim: 1024)
        embedding = outputs.pooler_output

        # Normalisation L2
        embedding = embedding / embedding.norm(
            p=2,
            dim=-1,
            keepdim=True
        )

    return embedding


# =========================================================
# Similarité DINOv2
# =========================================================

def dinov2_similarity(path1, path2):

    emb1 = get_dinov2_embedding(path1)
    emb2 = get_dinov2_embedding(path2)

    # Produit scalaire sur vecteurs normalisés = Similarité Cosinus
    similarity = torch.mm(emb1, emb2.T)

    return similarity.item()


# =========================================================
# Extraction des préfixes (Identique à votre logique)
# =========================================================

def extract_prefix_gt(fname):

    name = os.path.splitext(fname)[0]

    return re.split(
        r'_\d{2}[-_]\d{2}[-_]\d{2}\[\d{2}\]',
        name
    )[0]


def extract_prefix_method(fname):

    name = os.path.splitext(fname)[0]

    return name.split('_keyframe')[0]


# =========================================================
# Construction du dictionnaire
# =========================================================

def build_prefix_dict(folder, prefix_func):

    d = {}

    for fname in os.listdir(folder):

        if fname.lower().endswith(
            ('.png', '.jpg', '.jpeg')
        ):

            prefix = prefix_func(fname)

            d[prefix] = os.path.join(
                folder,
                fname
            )

    return d


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "gt_dir",
        help="Dossier ground truth"
    )

    parser.add_argument(
        "method_dir",
        help="Dossier méthode"
    )

    args = parser.parse_args()

    gt_dir = args.gt_dir
    method_dir = args.method_dir

    gt_images = build_prefix_dict(
        gt_dir,
        extract_prefix_gt
    )

    method_images = build_prefix_dict(
        method_dir,
        extract_prefix_method
    )

    # Changement du dossier de sortie pour différencier de CLIP
    output_dir = "Comparaison_DINOv2"
    os.makedirs(output_dir, exist_ok=True)

    output_name = (
        os.path.basename(
            os.path.normpath(method_dir)
        ) + ".txt"
    )

    output_path = os.path.join(
        output_dir,
        output_name
    )

    print("Taille GT :", len(gt_images))
    print("Taille Méthode :", len(method_images))

    with open(output_path, "w") as f:

        for prefix, gt_path in gt_images.items():

            if prefix in method_images:

                method_path = method_images[prefix]

                # Calcul de la similarité cosinus sémantique/géométrique
                sim = dinov2_similarity(
                    gt_path,
                    method_path
                )
                sim=1.-sim

                line = f"{prefix} {sim:.6f}\n"

                f.write(line)

                print(line.strip())

            else:

                print(
                    f"[WARNING] Pas d'image méthode pour : {prefix}"
                )

    print(f"\nRésultats écrits dans : {output_path}")