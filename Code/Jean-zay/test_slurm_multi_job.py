import argparse
import cv2


def compter_frames(video_path, output_path):
    # Chargement de la vidéo
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Erreur : Impossible d'ouvrir la vidéo '{video_path}'")
        return

    frame_count = 0

    # Parcours frame par frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Fin de la vidéo
        frame_count += 1

    cap.release()

    # Écriture du résultat dans le fichier de sortie
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"Nombre total de frames : {frame_count}\n")

    print(
        f"Traitement terminé. {frame_count} frames comptées et enregistrées dans '{output_path}'."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parcourt une vidéo frame par frame et enregistre le nombre total de frames."
    )

    # Arguments obligatoires
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Chemin vers le fichier vidéo d'entrée",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Chemin vers le fichier texte de sortie",
    )

    args = parser.parse_args()

    compter_frames(args.input, args.output)