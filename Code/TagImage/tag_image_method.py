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


def main():
    parser = argparse.ArgumentParser(description="Keyframe selection via spatially & temporally weighted label histograms")
    parser.add_argument('video', help='Path to input video file')
    parser.add_argument('--out', default='output', help='Output directory')
    parser.add_argument('--thr', type=float, default=0.7, help='Detection score threshold')
    parser.add_argument('--metric', default='euclidean', choices=['euclidean', 'cityblock', 'cosine'],
                        help='Distance metric to use')
    parser.add_argument('--sigma-pos', type=float, default=0.5,
                        help='Spatial weighting sigma (normalized distance)')
    parser.add_argument('--batch-size', type=int, default=None)
    args = parser.parse_args()

    #création du dossier s'il n'existe pas 
    os.makedirs(args.out, exist_ok=True)

    #extraction des frames
    cap = cv2.VideoCapture(args.video)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if num_frames == 0:
        print("Error: No frames in video.")
        return


    epsilon = 0.1 * num_frames

    # Préparation du modèle de détection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.batch_size:
        batch_size = args.batch_size
    else:
        batch_size = 4 if device.type == "cuda" else 1
    # Print qui permet de voir si le gpu est détecté
    print(batch_size)
    print("Device : ",device)

    model = models.detection.fasterrcnn_resnet50_fpn(
        weights=models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    ).to(device).eval()
    transform = T.ToTensor()

    # Nombre de classes du detecteur 
    num_classes = 91

    # Première passe
    histograms = []
    sharpness_scores = []

    batch_frames = []
    batch_imgs = []

    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    print("Computing weighted label histograms (batched)...")

    for i in tqdm(range(num_frames), desc="Histograms"):
        ret, frame = cap.read()
        if not ret:
            break

        # Stock frame pour sharpness
        batch_frames.append(frame)

        # Préparation image pour modèle
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        batch_imgs.append(transform(pil))

        # Si batch plein → inference
        if len(batch_imgs) == batch_size or i == num_frames - 1:

            imgs = [img.to(device) for img in batch_imgs]

            with torch.no_grad():
                outputs = model(imgs)

            for frame_b, out in zip(batch_frames, outputs):

                hist = compute_hist_from_output(
                    frame_b, out,
                    args.thr, num_classes,
                    args.sigma_pos
                )

                histograms.append(hist)
                sharpness_scores.append(compute_sharpness(frame_b))

            batch_frames = []
            batch_imgs = []

    cap.release()

    histograms = np.stack(histograms, axis=0)
    sharpness_scores = np.array(sharpness_scores)
    histograms = np.stack(histograms, axis=0)

    #normalisation qualité
    sharpness_scores = np.array(sharpness_scores)
    sharpness_scores /= sharpness_scores.max() 

    #Application de la pondération de qualité
    histograms *= sharpness_scores[:, None]  

    # Application de la pondération temporelle
    x = np.arange(num_frames, dtype=float)
    weights_t = double_gaussian_equal(x, num_frames, epsilon)
    weights_t /= weights_t.sum()    
    hist_t = np.average(histograms, axis=0, weights=weights_t)

    # calcul distance entre chaque histo et histo de référence
    dists = cdist(histograms, hist_t[None, :], metric=args.metric).squeeze()

    # Selection de la frame best
    top_idxs = np.argsort(dists)[:1]
    id_frame = top_idxs[0]


    # Sauvegarde des résultats
    base = os.path.splitext(os.path.basename(args.video))[0]
    out_path = os.path.join(args.out, f"{base}_keyframe__{id_frame:04d}.jpg")

    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, id_frame)
    ret, frame = cap.read()
    cap.release()

    if ret:
        cv2.imwrite(out_path, frame)



if __name__ == '__main__':
    main()
