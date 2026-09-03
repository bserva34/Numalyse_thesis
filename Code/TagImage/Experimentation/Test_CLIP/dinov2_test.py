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

def compare_images(path1, path2):

    emb1 = get_dinov2_embedding(path1)
    emb2 = get_dinov2_embedding(path2)

    # Produit scalaire sur vecteurs normalisés = Similarité Cosinus
    similarity = torch.mm(emb1, emb2.T)

    return 1.-similarity.item()

# ------------------------
# Tests
# ------------------------

score = compare_images(
    "ManOfTheWest_OurMethod.jpg",
    "ManOfTheWest_VV.png"
)

print(f"Our method : {score:.4f}")

score = compare_images(
    "ManOfTheWest_Pei.jpg",
    "ManOfTheWest_VV.png"
)

print(f"Pei: {score:.4f}")

score = compare_images(
    "ManOfTheWest_Zhang.jpg",
    "ManOfTheWest_VV.png"
)

print(f"Zhang : {score:.4f}")

score = compare_images(
    "ManOfTheWest_Middle.jpg",
    "ManOfTheWest_VV.png"
)

print(f"Middle : {score:.4f}")

score = compare_images(
    "ManOfTheWest_Chen.jpg",
    "ManOfTheWest_VV.png"
)

print(f"Chen : {score:.4f}")

score = compare_images(
    "ManOfTheWest_deb.png",
    "ManOfTheWest_VV.png"
)

print(f"Début : {score:.4f}")


score = compare_images(
    "johnwick3_keyframe_gt.jpg",
    "johnwick3_keyframe_gt.jpg"
)

print(f"Our Method - GT : {score:.4f}")


score = compare_images(
    "johnwick3_keyframe_yolo.jpg",
    "johnwick3_keyframe_gt.jpg"
)

print(f"Yolo - GT : {score:.4f}")

score = compare_images(
    "johnwick3_keyframe_retina.jpg",
    "johnwick3_keyframe_gt.jpg"
)

print(f"Retina - GT : {score:.4f}")

# score = compare_images(
#     "dinerdecons2_our.jpg",
#     "dinerdecons2_gt_unique.png"
# )

# print(f"Our - GT Unique : {score:.4f}")

# score = compare_images(
#     "dinerdecons2_rt-detr.jpg",
#     "dinerdecons2_gt_unique.png"
# )

# print(f"RT-DETR - GT Unique : {score:.4f}")

# score = compare_images(
#     "dinerdecons2_our.jpg",
#     "dinerdecons2_gt_q.jpg"
# )

# print(f"Our - GT Questionnaire : {score:.4f}")

# score = compare_images(
#     "dinerdecons2_rt-detr.jpg",
#     "dinerdecons2_gt_q.jpg"
# )

# print(f"RT-DETR - GT Questionnaire : {score:.4f}")