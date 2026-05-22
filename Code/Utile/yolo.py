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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Détection d'objets sur une vidéo avec YOLO et sauvegarde des frames annotées"
    )

    parser.add_argument(
        "video",
        help="Chemin vers la vidéo"
    )

    parser.add_argument(
        "--conf", "-c",
        type=float,
        default=0.5,
        help="Seuil de confiance minimal"
    )

    parser.add_argument(
        "--output", "-o",
        default="frames_output",
        help="Dossier de sortie pour les frames annotées"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Charger le modèle YOLO
    model = YOLO("yolov8n.pt")

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

        # Détection YOLO
        results = model(
            frame,
            conf=args.conf,
            verbose=False
        )

        # Copier l'image pour dessiner dessus
        annotated_frame = frame.copy()

        # Parcourir les résultats
        for r in results:

            boxes = r.boxes

            for box in boxes:
                # Coordonnées de la bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Classe détectée
                cls_id = int(box.cls[0])
                label = model.names[cls_id]

                # Score de confiance
                confidence = float(box.conf[0])

                # Texte affiché
                text = f"{label} {confidence:.2f}"

                # Dessiner la bounding box
                cv2.rectangle(
                    annotated_frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # Dessiner le label
                cv2.putText(
                    annotated_frame,
                    text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

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