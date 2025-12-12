import os
import torch
import numpy as np
from utils import get_frames, get_batches, visualize_predictions
from supernet_flattransf_3_8_8_8_13_12_0_16_60 import TransNetV2Supernet
from tqdm import tqdm

import matplotlib.pyplot as plt

def load_autoshot_model(ckpt_path="ckpt_0_200_0.pth"):
    """Charge le modèle AutoShot pré-entraîné."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    model = TransNetV2Supernet().eval().to(device)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Modèle non trouvé à {ckpt_path}")
    
    pretrained = torch.load(ckpt_path, map_location=device)
    model_dict = model.state_dict()
    pretrained_dict = pretrained["net"]
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print(f"Modèle chargé depuis {ckpt_path}")
    return model, device

def predict_video(model, device, video_path, threshold=0.296):
    """Retourne les frames où une coupure est détectée."""
    frames = get_frames(video_path)
    preds = []

    for batch in get_batches(frames):
        batch_tensor = torch.from_numpy(batch.transpose((3, 0, 1, 2))[np.newaxis, ...]).float().to(device)
        with torch.no_grad():
            out = model(batch_tensor)
            if isinstance(out, tuple):
                out = out[0]
            out = torch.sigmoid(out[0]).cpu().numpy()
        preds.append(out[25:75])  # partie utile du batch
    preds = np.concatenate(preds, 0)[:len(frames)]
    shot_boundaries = np.where(preds > threshold)[0]
    return shot_boundaries, preds, frames

def process_videos_in_directory(directory, output_dir,model,device):
    # Utilisez exist_ok=True pour éviter une exception si le dossier existe déjà
    os.makedirs(output_dir, exist_ok=True)
    
    video_files = [f for f in os.listdir(directory) if f.lower().endswith(('.mp4', '.avi', '.mkv'))]

    for video in tqdm(video_files):
        video_path = os.path.join(directory, video)
        video_basename = os.path.splitext(os.path.basename(video_path))[0]
        output_file = os.path.join(output_directory, f"{video_basename}.txt")
        if os.path.exists(output_file):
            print(f"Fichier de résultats {output_file} existe déjà")
            continue
        boundaries, preds, frames = predict_video(model, device, video_path)
        write_res(output_dir,video_path,boundaries,len(frames))


def write_res(directory,video_path,boundaries,lenght):
    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    output_file = os.path.join(directory, f"{video_basename}.txt")
    
    with open(output_file, "w") as f:
        i = 0
        while i < len(boundaries):
            b_start = boundaries[i]
            j = i + 1
            last_consecutive_boundary = b_start
            while j < len(boundaries) and boundaries[j] == boundaries[j-1] + 1:
                last_consecutive_boundary = boundaries[j]
                j += 1

            start = b_start 
            
            if last_consecutive_boundary > b_start:
                end = last_consecutive_boundary +1
            else:
                end = b_start +1
                
            f.write(f"{start}\t{end}\n")
            base = end
            i = j
    print(f"Résultats enregistrés dans {output_file}")    


if __name__ == "__main__":
    """execution unique"""
    import argparse
    parser = argparse.ArgumentParser(description='AutoShot detector')
    parser.add_argument('video', help='chemin vers la vidéo mp4/avi')
    parser.add_argument('--directory', help='chemin du dossier de sortie', default=None)
    parser.add_argument('--v', action='store_true')
    args = parser.parse_args()

    model, device = load_autoshot_model()

    boundaries, preds, frames = predict_video(model, device, args.video)
    print("Coupures détectées :", boundaries)

    if args.directory : 
        write_res(args.directory,args.video,boundaries,len(frames))

    # Visualisation
    if args.v:
        output_dir = "results_autoshot"
        os.makedirs(output_dir, exist_ok=True)
        vis = visualize_predictions(frames, predictions=(preds > 0.296).astype(np.uint8))
        vis.save(os.path.join(output_dir, os.path.basename(args.video).replace(".mp4", "_shots.png")))
        print(f"Résultat sauvegardé : {output_dir}")

    """exuction multiple sur dossier"""
    # model, device = load_autoshot_model()

    # input_directory = r"../../../Dataset/Dataset_Shot/BBC/videos"  # Dossier source de vos vidéos
    # output_directory = r"../Experimentation/AutoShot_BBC"  # Dossier de sortie
    
    # process_videos_in_directory(input_directory, output_directory,model,device)
