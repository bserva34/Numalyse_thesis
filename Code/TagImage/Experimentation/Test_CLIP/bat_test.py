import os
import re
import argparse
import cv2

from PIL import Image



def compare_images(path1, path2):
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