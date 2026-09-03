import os
import re
import argparse

from PIL import Image

import torch
from transformers import CLIPProcessor, CLIPModel


# =========================================================
# Chargement modèle CLIP
# =========================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained(
    "openai/clip-vit-large-patch14"
).to(device)

processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-large-patch14"
)

model.eval()


# =========================================================
# Embedding image
# =========================================================

def get_image_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        # 1. On passe par le modèle de vision de base
        vision_outputs = model.vision_model(**inputs)
        
        # 2. On récupère la sortie poolée (l'embedding visuel brut)
        pooled_output = vision_outputs.pooler_output  # Dim: [1, 1024] pour ViT-L
        
        # 3. ON PROJETTE dans l'espace sémantique multimodal de CLIP (Crucial !)
        image_features = model.visual_projection(pooled_output)  # Dim: [1, 768]

        # 4. Normalisation L2 pour la similarité cosinus
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

    return image_features

def clip_similarity(path1, path2):
    emb1 = get_image_embedding(path1)
    emb2 = get_image_embedding(path2)

    # Si les vecteurs sont déjà normalisés L2, le produit scalaire (dot product) 
    # est strictement égal à la similarité cosinus.
    similarity = torch.mm(emb1, emb2.T)

    return similarity.item()


# =========================================================
# Prefix extraction
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
# Build dict
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

    output_dir = "Comparaison_methode_CLIP"
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

                sim = clip_similarity(
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