from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

# ------------------------
# Device
# ------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------
# Chargement modèle
# ------------------------

model = CLIPModel.from_pretrained(
    "openai/clip-vit-large-patch14"
).to(device)

processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-large-patch14"
)

model.eval()

# ------------------------
# Embedding image
# ------------------------

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

# ------------------------
# Similarité
# ------------------------

def compare_images(path1, path2):
    emb1 = get_image_embedding(path1)
    emb2 = get_image_embedding(path2)

    # Si les vecteurs sont déjà normalisés L2, le produit scalaire (dot product) 
    # est strictement égal à la similarité cosinus.
    similarity = torch.mm(emb1, emb2.T)

    return 1.-similarity.item()

# ------------------------
# Tests
# ------------------------

# score = compare_images(
#     "ManOfTheWest_OurMethod.jpg",
#     "ManOfTheWest_VV.png"
# )

# print(f"Our method : {score:.4f}")

# score = compare_images(
#     "ManOfTheWest_Pei.jpg",
#     "ManOfTheWest_VV.png"
# )

# print(f"Pei: {score:.4f}")

# score = compare_images(
#     "ManOfTheWest_Zhang.jpg",
#     "ManOfTheWest_VV.png"
# )

# print(f"Zhang : {score:.4f}")

# score = compare_images(
#     "ManOfTheWest_Middle.jpg",
#     "ManOfTheWest_VV.png"
# )

# print(f"Middle : {score:.4f}")

# score = compare_images(
#     "ManOfTheWest_Chen.jpg",
#     "ManOfTheWest_VV.png"
# )

# print(f"Chen : {score:.4f}")

# score = compare_images(
#     "ManOfTheWest_deb.png",
#     "ManOfTheWest_VV.png"
# )

# print(f"Début : {score:.4f}")


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