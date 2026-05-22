import cv2
import math
import os
from PIL import Image


def video_to_contact_sheet(
    video_path,
    output_path="frames_sheet.jpg",
    frame_step=1,
    frames_per_row=25,
    resize_width=None
):
    """
    Génère une image contenant des frames concaténées d'une vidéo.

    Paramètres :
    - video_path : chemin de la vidéo
    - output_path : chemin de sortie de l'image finale
    - frame_step : prend 1 frame sur N
        ex:
            1 = toutes les frames
            2 = une frame sur deux
            5 = une frame sur cinq
    - frames_per_row : nombre de frames par ligne
    - resize_width : largeur des miniatures (None = taille originale)
    """

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise Exception(f"Impossible d'ouvrir la vidéo : {video_path}")

    frames = []
    frame_index = 0
    saved_count = 0

    print("Extraction des frames...")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Garde seulement une frame selon l'intervalle
        if frame_index % frame_step == 0:

            # OpenCV -> RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            img = Image.fromarray(frame_rgb)

            # Redimensionnement optionnel
            if resize_width is not None:
                ratio = resize_width / img.width
                new_height = int(img.height * ratio)

                img = img.resize(
                    (resize_width, new_height),
                    Image.Resampling.LANCZOS
                )

            frames.append(img)
            saved_count += 1

            print(f"Frame ajoutée : {saved_count}")

        frame_index += 1

    cap.release()

    if len(frames) == 0:
        raise Exception("Aucune frame extraite.")

    # Taille d'une miniature
    thumb_width, thumb_height = frames[0].size

    # Calcul du nombre de lignes
    rows = math.ceil(len(frames) / frames_per_row)

    # Taille de l'image finale
    sheet_width = frames_per_row * thumb_width
    sheet_height = rows * thumb_height

    # Création du canvas final
    sheet = Image.new("RGB", (sheet_width, sheet_height), color=(0, 0, 0))

    print("Assemblage des frames...")

    for i, frame in enumerate(frames):
        x = (i % frames_per_row) * thumb_width
        y = (i // frames_per_row) * thumb_height

        sheet.paste(frame, (x, y))

    # Sauvegarde
    sheet.save(output_path)

    print(f"\nImage sauvegardée : {output_path}")
    print(f"Nombre total de frames : {len(frames)}")


if __name__ == "__main__":

    video_to_contact_sheet(
        video_path="adele1_extrait.mp4",
        output_path="mosaic.jpg",

        # 1 = toutes les frames
        # 2 = une sur deux
        # 5 = une sur cinq
        frame_step=1,

        # 25 images par ligne
        frames_per_row=15,

        # largeur des miniatures
        resize_width=480
    )