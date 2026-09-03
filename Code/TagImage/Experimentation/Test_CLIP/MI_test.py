import os
import re
import argparse
import cv2
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

print("--------------------------------")

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

print("--------------------------------")

score = compare_images(
    "dinerdecons2_our.jpg",
    "dinerdecons2_gt_unique.png"
)

print(f"Our - GT Unique : {score:.4f}")

score = compare_images(
    "dinerdecons2_rt-detr.jpg",
    "dinerdecons2_gt_unique.png"
)

print(f"RT-DETR - GT Unique : {score:.4f}")

score = compare_images(
    "dinerdecons2_our.jpg",
    "dinerdecons2_gt_q.jpg"
)

print(f"Our - GT Questionnaire : {score:.4f}")

score = compare_images(
    "dinerdecons2_rt-detr.jpg",
    "dinerdecons2_gt_q.jpg"
)

print(f"RT-DETR - GT Questionnaire : {score:.4f}")