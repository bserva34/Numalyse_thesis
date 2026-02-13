import os
from scenedetect import open_video, SceneManager, FrameTimecode
from scenedetect.detectors import AdaptiveDetector, ContentDetector, ThresholdDetector, HistogramDetector, HashDetector

from scenedetect.video_stream import VideoOpenFailure


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
    detector = detector_class()
    scene_manager.add_detector(detector)
    
    try:
        scene_manager.detect_scenes(video, show_progress=False)
    except Exception as e:
        print(f"Erreur lors du traitement de {video_path} avec {detector_class.__name__} : {e}")
        return
    
    scene_list = scene_manager.get_scene_list()
    
    with open(output_file, "w") as f:
        for scene in scene_list:
            start, end = scene
            f.write(f"{start.get_frames()}\t{end.get_frames()-1}\n")
    
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
    input_directory = r"../../../Dataset/Dataset_Shot/AutoShot/video" 
    output_directory = r"../Experimentation/AutoShot_TEST/Adaptive_AutoShot"  # Dossier de sortie
    
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
