"""
Prérequis:
    pip install torch torchvision opencv-python

Usage:
    python3 detect_and_save_frames.py chemin/vers/video.mp4
"""

import argparse
import cv2
import os
import torch

from torchvision import models
from torchvision.transforms import transforms as T


# Classes COCO
COCO_CLASSES = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    "N/A", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "N/A", "backpack", "umbrella", "N/A", "N/A", "handbag", "tie",
    "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "N/A", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "N/A", "dining table", "N/A",
    "N/A", "toilet", "N/A", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "N/A", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Détection d'objets avec Faster R-CNN et sauvegarde des frames annotées"
    )

    parser.add_argument(
        "video",
        help="Chemin vers la vidéo"
    )

    parser.add_argument(
        "--conf", "-c",
        type=float,
        default=0.75,
        help="Seuil de confiance minimal"
    )

    parser.add_argument(
        "--output", "-o",
        default="frames_output",
        help="Dossier de sortie"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] Utilisation du device: {device}")

    # Charger le modèle Faster R-CNN
    # model = models.detection.fasterrcnn_resnet50_fpn(
    #     weights=models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    # ).to(device).eval()


    # Charger le modèle RetinaNet
    model = models.detection.retinanet_resnet50_fpn(
        weights=models.detection.RetinaNet_ResNet50_FPN_Weights.DEFAULT
    ).to(device).eval()

    # Transformation image -> tensor
    transform = T.ToTensor()

    # Créer dossier sortie
    os.makedirs(args.output, exist_ok=True)

    # Ouvrir vidéo
    cap = cv2.VideoCapture(args.video)

    if not cap.isOpened():
        print(f"Erreur: impossible d'ouvrir la vidéo {args.video}")
        return

    frame_idx = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # OpenCV -> RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Tensor
        image_tensor = transform(rgb_frame).to(device)

        # Inference
        with torch.no_grad():
            outputs = model([image_tensor])

        output = outputs[0]

        annotated_frame = frame.copy()

        boxes = output["boxes"]
        labels = output["labels"]
        scores = output["scores"]

        for box, label, score in zip(boxes, labels, scores):

            confidence = score.item()

            if confidence < args.conf:
                continue

            x1, y1, x2, y2 = map(int, box.tolist())

            cls_id = int(label.item())
            class_name = COCO_CLASSES[cls_id]

            text = f"{class_name} {confidence:.2f}"

            # Bounding box
            cv2.rectangle(
                annotated_frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Label
            cv2.putText(
                annotated_frame,
                text,
                (x1, max(y1 - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        # Sauvegarde frame
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