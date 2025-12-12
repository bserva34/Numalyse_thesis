import os
from scenedetect import open_video, SceneManager, FrameTimecode
from scenedetect.detectors import AdaptiveDetector, ContentDetector, ThresholdDetector, HistogramDetector, HashDetector

def process_video_with_detector(video_path, detector_class, output_dir):
    print(f"Traitement de {video_path} avec {detector_class.__name__}...")
    
    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    output_file = os.path.join(output_dir, f"{video_basename}_{detector_class.__name__}.txt")

    if os.path.exists(output_file):
        print(f"Fichier de résultats {output_file} existe déjà. Passage à la vidéo/détecteur suivant.")
        return

    video = open_video(video_path)
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
            f.write(f"{start.get_frames()}\t{end.get_frames()}\n")
    
    print(f"Résultats enregistrés dans {output_file}")

def process_videos_in_directory(directory, output_dir):
    # Utilisez exist_ok=True pour éviter une exception si le dossier existe déjà
    os.makedirs(output_dir, exist_ok=True)
    
    video_files = [f for f in os.listdir(directory) if f.lower().endswith(('.mp4', '.avi', '.mkv'))]
    #detector_classes = [AdaptiveDetector, ContentDetector, ThresholdDetector, HistogramDetector, HashDetector]
    detector_classes = [AdaptiveDetector]

    for video in video_files:
        video_path = os.path.join(directory, video)
        for detector_class in detector_classes:
            process_video_with_detector(video_path, detector_class, output_dir)

if __name__ == "__main__":
    input_directory = r"../../../Dataset/Dataset_Shot/ClipShots/videos/merged"  # Dossier source de vos vidéos
    output_directory = r"../Experimentation/AdaptiveDetector"  # Dossier de sortie
    
    process_videos_in_directory(input_directory, output_directory)
