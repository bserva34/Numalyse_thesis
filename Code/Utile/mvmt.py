import cv2
import numpy as np
import os

video_path = "v.mp4"
output_dir = "motion_red"

os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)

ret, prev_frame = cap.read()
if not ret:
    raise ValueError("Impossible de lire la vidéo")

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

frame_idx = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calcul optical flow
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )

    # magnitude du mouvement
    magnitude, _ = cv2.cartToPolar(
        flow[..., 0],
        flow[..., 1]
    )

    # normalisation 0-255
    motion_intensity = cv2.normalize(
        magnitude,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    motion_intensity = motion_intensity.astype(np.uint8)

    # image rouge
    red_mask = np.zeros_like(frame)

    # canal rouge uniquement (BGR -> index 2)
    red_mask[..., 2] = motion_intensity

    # fusion avec image originale
    overlay = cv2.addWeighted(
        frame,
        0.7,
        red_mask,
        1.0,
        0
    )

    # sauvegarde
    out_path = os.path.join(
        output_dir,
        f"frame_{frame_idx:05d}.png"
    )

    cv2.imwrite(out_path, overlay)

    prev_gray = gray
    frame_idx += 1

cap.release()

print("Terminé.")