import os
from scenedetect import open_video, SceneManager, FrameTimecode
from scenedetect.detectors import AdaptiveDetector, ContentDetector, ThresholdDetector, HistogramDetector, HashDetector

import cv2
import numpy as np

from scenedetect.video_stream import VideoOpenFailure

from skimage.metrics import structural_similarity as ssim

def ssim_distance(f1, f2):
    gray1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(gray1, gray2, full=True)
    return 1 - score

def bhattacharyya_distance(f1, f2):
    h1 = cv2.calcHist([f1], [0,1,2], None, [8,8,8], [0,256]*3)
    h2 = cv2.calcHist([f2], [0,1,2], None, [8,8,8], [0,256]*3)
    cv2.normalize(h1, h1)
    cv2.normalize(h2, h2)
    return cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA)

def get_frame(cap, frame_id):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = cap.read()
    if not ret:
        return None
    return frame

def get_avg_frame(cap, frame_id, window=2):
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in range(-window, window + 1):
        fid = frame_id + i
        if fid < 0 or fid >= total_frames:
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        if ret:
            frames.append(frame.astype(np.float32))

    if len(frames) == 0:
        return None

    return np.mean(frames, axis=0).astype(np.uint8)

def pixel_diff(f1, f2):
    return cv2.norm(f1, f2, cv2.NORM_L1) / (f1.shape[0] * f1.shape[1])

def compute_score(f1, f2):
    h = bhattacharyya_distance(f1, f2)
    p = pixel_diff(f1, f2)
    s = ssim_distance(f1, f2)

    return 0.4 * h + 0.3 * (p / 255.0) + 0.3 * s

def merge_scenes(video_path, scenes, threshold):

    cap = cv2.VideoCapture(video_path)

    merged = []
    current_start, current_end = scenes[0]

    for i in range(1, len(scenes)):

        next_start, next_end = scenes[i]

        last_frame_prev = get_frame(cap, current_end)
        first_frame_next = get_frame(cap, next_start)
        # last_frame_prev = get_avg_frame(cap, current_end, window=2)
        # first_frame_next = get_avg_frame(cap, next_start, window=2)

        d = compute_score(last_frame_prev, first_frame_next)

        if d < threshold:
            # fusion
            current_end = next_end
            # name1 = "Save_img_merged/frame_"+str(current_end)+".png"
            # name2 = "Save_img_merged/frame_"+str(next_start)+".png"
            # cv2.imwrite(name1, last_frame_prev)
            # cv2.imwrite(name2, first_frame_next)
        else:
            # name1 = "Save_img_merged/frame_"+str(current_end)+"_fin.png"
            # name2 = "Save_img_merged/frame_"+str(next_start)+"_deb.png"
            # cv2.imwrite(name1, last_frame_prev)
            # cv2.imwrite(name2, first_frame_next)
            merged.append((current_start, current_end))
            current_start, current_end = next_start, next_end

    merged.append((current_start, current_end))

    cap.release()
    return merged


def process_video_with_detector(video_path, detector_class, output_dir):
    #print(f"Traitement de {video_path} avec {detector_class.__name__}...")
    
    video_basename = os.path.splitext(os.path.basename(video_path))[0]

    #output_file = os.path.join(output_dir, f"{video_basename}_{detector_class.__name__}.txt")
    output_file = os.path.join(output_dir, f"{video_basename}.txt")

    if os.path.exists(output_file):
        #print(f"Fichier de résultats {output_file} existe déjà. Passage à la vidéo/détecteur suivant.")
        return

    try:
        video = open_video(video_path)
    except VideoOpenFailure as e:
        print(f"[ERREUR] Impossible d'ouvrir la vidéo : {video_path}")
        print(f"         {e}")
        return  # on passe à la vidéo suivante
    scene_manager = SceneManager()
    scene_manager.auto_downscale = True
    #detector = detector_class()
    detector = AdaptiveDetector(adaptive_threshold=3.0)
    scene_manager.add_detector(detector)
    
    try:
        scene_manager.detect_scenes(video, show_progress=False)
    except Exception as e:
        print(f"Erreur lors du traitement de {video_path} avec {detector_class.__name__} : {e}")
        return
    
    scene_list = scene_manager.get_scene_list()

    scenes = [(s[0].get_frames(), s[1].get_frames()-1) for s in scene_list]
    #scenes = merge_scenes(video_path, scenes, threshold=0.15)
    with open(output_file, "w") as f:
        for start, end in scenes:
            f.write(f"{start}\t{end}\n")
    
    # with open(output_file, "w") as f:
    #     for scene in scene_list:
    #         start, end = scene
    #         f.write(f"{start.get_frames()}\t{end.get_frames()-1}\n")
    
    print(f"Résultats enregistrés dans {output_file}")

def process_videos_in_directory(directory, output_dir):
    # Utilisez exist_ok=True pour éviter une exception si le dossier existe déjà
    os.makedirs(output_dir, exist_ok=True)
    
    video_files = [f for f in os.listdir(directory) if f.lower().endswith(('.mp4', '.avi', '.mkv', '.mov', '.m4v', '.mpe'))]
    #detector_classes = [AdaptiveDetector, ContentDetector, ThresholdDetector, HistogramDetector, HashDetector]
    detector_classes = [AdaptiveDetector]
    if len(video_files) == 0 : print("problème lecture : ",directory)
    for video in video_files:
        video_path = os.path.join(directory, video)
        for detector_class in detector_classes:
            process_video_with_detector(video_path, detector_class, output_dir)

if __name__ == "__main__":
    # input_directory = r"../../../Dataset/Dataset_Shot/AutoShot/video" 
    # output_directory = r"../Experimentation/AutoShot_TEST/Adaptive_AutoShot"  # Dossier de sortie

    # input_directory = r"../../../Dataset/Dataset_Shot/ClipShots/videos/merged" 
    # output_directory = r"../Experimentation/ClipShots_TEST/Adaptive"  # Dossier de sortie

    # input_directory = r"../../../Dataset/Dataset_Shot/BBC/BBC_base" 
    # output_directory = r"../Experimentation/BBC_TEST/Adaptive_PT"  # Dossier de sortie

    input_directory = r"../../../Dataset/Dataset_Shot/CineShots/videos" 
    output_directory = r"../Experimentation/TEST_CUT_DETECTION/Adaptive"  # Dossier de sortie
    
    process_videos_in_directory(input_directory, output_directory)

    # import argparse
    # parser = argparse.ArgumentParser(description='AutoShot detector')
    # parser.add_argument('list', help='chemin list')
    # args = parser.parse_args()

    # input_directory = r"../../../Dataset/Dataset_Shot/V3C/V3C1/videos"  # Dossier source de vos vidéos
    # #list_of_videos = r"../../../Dataset/Dataset_Shot/V3C/V3C1/echantillons/echantillon_2.txt"

    # ech_basename = os.path.splitext(os.path.basename(args.list))[0]

    # output_directory = r"../Experimentation/Adaptive_V3C1"  # Dossier de sortie
    # output_directory = output_directory+"_"+ech_basename

    # with open(args.list, "r", encoding="utf-8") as f:
    #     for line in f:
    #         line = line.strip()  # enlève \n et les espaces
    #         #print("Traitement de la vidéo : ",line)
    #         dir_prov=os.path.join(input_directory, line)
    #         process_videos_in_directory(dir_prov, output_directory)
