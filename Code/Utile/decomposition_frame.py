"""
Prérequis:
    pip install ultralytics opencv-python

Usage:
    python3 detect_and_save_frames.py chemin/vers/video.mp4
"""

import argparse
import cv2
import os
from ultralytics import YOLO

from ultralytics import RTDETR


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "video",
        help="Chemin vers la vidéo"
    )

    parser.add_argument(
        "--output", "-o",
        default="frames_output",
        help="Dossier de sortie pour les frames"
    )

    return parser.parse_args()


def main():
    args = parse_args()


    # Créer le dossier de sortie
    os.makedirs(args.output, exist_ok=True)

    # Ouvrir la vidéo
    cap = cv2.VideoCapture(args.video)

    if not cap.isOpened():
        print(f"Erreur: impossible d'ouvrir la vidéo {args.video}")
        return

    frame_idx = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Copier l'image pour dessiner dessus
        annotated_frame = frame.copy()

        # Sauvegarder la frame annotée
        output_path = os.path.join(
            args.output,
            f"frame_{frame_idx:06d}.jpg"
        )

        cv2.imwrite(output_path, annotated_frame)

        print(f"[INFO] Frame sauvegardée: {output_path}")

        frame_idx += 1

    cap.release()

    print("\nTraitement terminé.")
    print(f"Frames enregistrées dans: {args.output}")


if __name__ == "__main__":
    main()