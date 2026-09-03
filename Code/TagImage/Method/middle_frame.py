import cv2
import os
import argparse

def extract_middle_frame(video_path, out_dir='keyframes'):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    os.makedirs(out_dir, exist_ok=True)
    
    # Correction : Extraction propre du nom sans dépendre de la variable globale 'args'
    name_of_video = os.path.splitext(os.path.basename(video_path))[0]

    middle_index = total_frames // 2
    ret = False
    frame = None

    # Première tentative : Saut direct au milieu théorique
    if middle_index > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_index)
        ret, frame = cap.read()

    # Sécurité / Correction : Si le saut a échoué (index théorique faux ou corrompu)
    if not ret:
        print(f"⚠️ Index théorique invalide pour {name_of_video}. Recherche du vrai milieu...")
        
        # On cherche la fin réelle en reculant à partir du total théorique
        actual_total = total_frames
        while actual_total > 0:
            actual_total -= 10  # On recule par pas de 10 pour aller vite
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, actual_total))
            test_ret, _ = cap.read()
            if test_ret:
                break
                
        # Si on a trouvé une zone lisible, on recalcule le milieu réel
        if actual_total > 0:
            middle_index = actual_total // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, middle_index)
            ret, frame = cap.read()

    # Sauvegarde si une frame a pu être récupérée
    if ret and frame is not None:
        fname = os.path.join(out_dir, f"{name_of_video}_keyframe_{middle_index:04d}.jpg")
        cv2.imwrite(fname, frame)
        print(f"✓ Frame du milieu ({middle_index}) extraite pour {name_of_video}")
    else:
        print(f"❌ Impossible de lire la vidéo {name_of_video} (fichier totalement illisible).")

    cap.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extraction de la frame du milieu')
    parser.add_argument('video', help='Chemin vers la vidéo')
    parser.add_argument('--out', default='keyframes', help='Répertoire de sortie')
    args = parser.parse_args()

    extract_middle_frame(args.video, out_dir=args.out)