import os
import argparse
import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
import torchvision.models as models
from tqdm import tqdm
from scipy.spatial.distance import cdist

import matplotlib.pyplot as plt

from ultralytics import RTDETR
from ultralytics import YOLO


# Pondération Temporel 
def double_gaussian_equal(x, t, epsilon):
    """
    t : nombre de frames / longueurs de plans
    epsilon : 0.1*t / décalage de la deuxième gaussienne 
    sigma : largeur de chaque gaussienne 
    """ 
    sigma = 0.15 * t
    A = 1.0
    g1 = A * np.exp(-((x - t/2)**2) / (2 * sigma**2))
    g2 = A * np.exp(-((x - (t - epsilon))**2) / (2 * sigma**2))
    return g1 + g2

# Pondération de Qualité
def compute_sharpness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def compute_hist_from_output(frame, out, thr, num_classes, sigma_pos):

    # Extract detections
    scores = out['scores'].cpu().numpy()
    labels = out['labels'].cpu().numpy()
    boxes = out['boxes'].cpu().numpy()

    # Filter by threshold
    mask = scores > thr
    scores_f = scores[mask]
    labels_f = labels[mask]
    boxes_f = boxes[mask]

    # Image dimensions
    H, W = frame.shape[:2]
    cx_img, cy_img = W / 2.0, H / 2.0

    # Center of boxes
    cx = (boxes_f[:, 0] + boxes_f[:, 2]) / 2.0
    cy = (boxes_f[:, 1] + boxes_f[:, 3]) / 2.0

    # Distance to image center (normalized)
    dx = (cx - cx_img) / cx_img
    dy = (cy - cy_img) / cy_img
    dist = np.sqrt(dx**2 + dy**2)

    # Pondération spatial
    w_pos = np.exp(-0.5 * (dist / sigma_pos)**2)
    weights = scores_f * w_pos
    
    # Création de l'histogramme
    hist = np.zeros(num_classes, dtype=np.float32)
    np.add.at(hist, labels_f, weights)

    return hist


def process_video(
    model,
    video_path,
    out_dir,
    thr=0.8,
    metric="euclidean",
    sigma_pos=0.5,
    batch_size=None
):

    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if num_frames == 0:
        print(f"Erreur : {video_path} ne contient aucune frame")
        return 0

    video_duration = num_frames / fps if fps > 0 else 0

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    if batch_size is None:
        batch_size = 16 if device.type == "cuda" else 1

    num_classes = 80

    histograms = []
    sharpness_scores = []

    batch_frames = []
    batch_imgs = []

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    print(f"\nTraitement : {os.path.basename(video_path)}")

    for i in tqdm(range(num_frames), desc="Frames"):

        ret, frame = cap.read()

        if not ret:
            break

        batch_frames.append(frame)
        batch_imgs.append(frame)

        if len(batch_imgs) == batch_size or i == num_frames - 1:

            with torch.inference_mode():

                outputs = model(
                    batch_imgs,
                    device=device,
                    verbose=False
                )

            for frame_b, out in zip(batch_frames, outputs):

                yolo_out = {
                    "scores": out.boxes.conf,
                    "labels": out.boxes.cls.long(),
                    "boxes": out.boxes.xyxy
                }

                hist = compute_hist_from_output(
                    frame_b,
                    yolo_out,
                    thr,
                    num_classes,
                    sigma_pos
                )

                histograms.append(hist)

                sharpness_scores.append(
                    compute_sharpness(frame_b)
                )

            batch_frames.clear()
            batch_imgs.clear()

    cap.release()

    if len(histograms) == 0:
        print(f"Erreur : Aucune frame n'a pu être lue dans {video_path}")
        return video_duration

    # Conversion en tableau NumPy
    histograms = np.stack(histograms)
    
    # ─── MODIFICATION ICI : Calcul basé sur le nombre de frames réellement lues ───
    actual_num_frames = len(histograms)

    sharpness_scores = np.array(
        sharpness_scores,
        dtype=np.float32
    )

    if sharpness_scores.max() > 0:
        sharpness_scores /= sharpness_scores.max()

    histograms *= sharpness_scores[:, None]

    # Génération de la rampe temporelle et d'epsilon selon les frames réelles
    x = np.arange(actual_num_frames, dtype=float)
    actual_epsilon = 0.1 * actual_num_frames

    weights_t = double_gaussian_equal(
        x,
        actual_num_frames,
        actual_epsilon
    )

    weights_t /= weights_t.sum()

    # Les dimensions de histograms (axis=0) et de weights_t concordent désormais à 100%
    hist_t = np.average(
        histograms,
        axis=0,
        weights=weights_t
    )
    # ─────────────────────────────────────────────────────────────────────────────

    dists = cdist(
        histograms,
        hist_t[None, :],
        metric=metric
    ).squeeze()

    # Gestion du cas particulier où une seule frame a été extraite
    if dists.ndim == 0:
        id_frame = 0
    else:
        id_frame = np.argmin(dists)

    cap = cv2.VideoCapture(video_path)

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        int(id_frame)
    )

    ret, frame = cap.read()

    cap.release()

    if ret:

        base = os.path.splitext(
            os.path.basename(video_path)
        )[0]

        out_path = os.path.join(
            out_dir,
            f"{base}_keyframe_{id_frame:04d}.jpg"
        )

        cv2.imwrite(out_path, frame)

    return video_duration

if __name__ == '__main__':
    main()
