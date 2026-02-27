import cv2
import torch
import numpy as np
import joblib
import os

from c3d_sbd import C3D_SBD


def create_segments(video_path, length_segment=8, overlap=4):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Impossible d'ouvrir la vidéo: {video_path}")
    except Exception as e:
        print(f"[ERREUR] {e}")
        return []        

    frames = []
    while True:
        ret, fr = cap.read()
        if not ret:
            break
        fr = cv2.resize(fr, (112, 112))
        frames.append(fr)
    cap.release()

    #print(len(frames))

    segments = []
    step = length_segment - overlap
    for i in range(0, len(frames) - length_segment + 1, step):
        segments.append({
            "frames": np.array(frames[i:i+length_segment]),
            "start": i,
            "end": i + length_segment - 1
        })

    return segments


# def extract_c3d_features(segment_frames, model, device="cuda"):
#     x = torch.from_numpy(segment_frames).float()
#     x = x.permute(3,0,1,2).unsqueeze(0) / 255.0
#     x = x.to(device)
 
#     with torch.no_grad():
#         feat = model(x, return_features=True)

#     return feat.cpu().numpy().squeeze()

def extract_c3d_features_batch(segments, model, device="cuda", batch_size=64):
    features = []

    for i in range(0, len(segments), batch_size):
        batch = segments[i:i+batch_size]

        x = torch.stack([
            torch.from_numpy(s["frames"]).float().permute(3,0,1,2)
            for s in batch
        ]) / 255.0

        x = x.to(device)

        with torch.no_grad():
            feat = model(x, return_features=True)

        features.append(feat.cpu())

    return torch.cat(features, dim=0).numpy()



def Merge_res(results): 
    if not results: 
        return [] 
    merged = [results[0]] 
    for r in results[1:]: 
        if r["label"] == merged[-1]["label"]: 
            merged[-1]["end"] = r["end"] 
        else: merged.append(r) 
    return merged

# def Merge_res(results):
#     if not results:
#         return []

#     results = sorted(results, key=lambda x: x["start"])

#     merged = []
#     buffer = [results[0]]

#     for r in results[1:]:
#         print(r["end"])
#         if r["label"] == buffer[-1]["label"]:
#             print("same")
#             buffer.append(r)
#         else:
#             print("diff")
#             seg = _merge_buffer_safe(buffer)
#             if seg is not None:
#                 merged.append(seg)
#             buffer = [r]

#     #print("fin boucle",len(buffer))
#     seg = _merge_buffer_safe(buffer)
#     if seg is not None:
#         seg["end"]+=1
#         merged.append(seg)

#     return merged



# def _merge_buffer_safe(buffer):
#     if not buffer:
#         return None

#     # if len(buffer) == 1:
#     #     return buffer[0].copy()

#     label = buffer[0]["label"]

#     if label == 0:
#         start = buffer[0]["start"] 
#         if start>0: start = start+1
#         end = max(b["end"] for b in buffer)-1
#         print(label,start,end)
#     else:
#         start = max(b["start"] for b in buffer)-1
#         end = min(b["end"] for b in buffer)+1
#         print(label,start,end)

#     if end <= start:
#         return None

#     frames = []
#     for b in buffer:
#         frames.append(b["frames"])
#     frames = np.concatenate(frames, axis=0)

#     out = buffer[0].copy()
#     out["start"] = start
#     out["end"] = end
#     out["frames"] = frames

#     return out





def bhattacharyya_distance(f1, f2):
    h1 = cv2.calcHist([f1], [0,1,2], None, [8,8,8], [0,256]*3)
    h2 = cv2.calcHist([f2], [0,1,2], None, [8,8,8], [0,256]*3)
    cv2.normalize(h1, h1)
    cv2.normalize(h2, h2)
    return cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA)


def Post_Processing(segments, threshold=0.3):
    for s in segments:
        if s["label"] == 0:
            continue
        d = bhattacharyya_distance(s["first_frame"], s["last_frame"])
        if d < threshold:
            s["label"] = 0
    return segments


def SBD_detect(video_path, model, svm, batch_size=32):
    segments = create_segments(video_path)

    feats = extract_c3d_features_batch(segments, model, batch_size=batch_size)

    labels = svm.predict(feats)
    scores = svm.predict_proba(feats)

    results = []
    for s, label, score in zip(segments, labels, scores):
        results.append({
            "start": s["start"],
            "end": s["end"],
            "label": label,
            "score": score,
            "first_frame": s["frames"][0],
            "last_frame": s["frames"][-1]
        })

    merged = Merge_res(results)
    final = Post_Processing(merged)

    #display_res(final)

    return final



def display_res(res):
    for r in res:
        lbl = ["NO", "SHARP", "GRADUAL"][r["label"]]
        print(f"[{r['start']:6d} → {r['end']:6d}] : {lbl}")

def write_res(output_file,res):
    with open(output_file, "w") as f:
        for r in res:
            f.write(f"{r['start']:6d}\t{r['end']:6d} : {r['label']}\n")
    print(f"Résultats enregistrés dans {output_file}")


def process_videos_in_directory(directory, output_dir,model,svm):
    # Utilisez exist_ok=True pour éviter une exception si le dossier existe déjà
    os.makedirs(output_dir, exist_ok=True)
    
    video_files = [f for f in os.listdir(directory) if f.lower().endswith(('.mp4', '.avi', '.mkv', '.mov', '.m4v', '.mpe'))]

    if len(video_files) == 0 : print("problème lecture : ",directory)
    for video in video_files:
        video_path = os.path.join(directory, video)
        video_basename = os.path.splitext(os.path.basename(video_path))[0]
        output_file = os.path.join(output_dir, f"{video_basename}.txt")
        if os.path.exists(output_file):
            print(f"Fichier de résultats {output_file} existe déjà")
            continue
        else :
            print("Traitement de la video : ",video_basename)
            res=SBD_detect(video_path, model, svm)
            write_res(output_file,res)
            del res

if __name__ == "__main__":
    # import argparse

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--video", required=True)
    # args = parser.parse_args()

    # device = "cuda" if torch.cuda.is_available() else "cpu"

    # model = C3D_SBD().to(device)
    # model.load_state_dict(torch.load("weights_save/c3d_sbd_best.pth", map_location=device))
    # model.eval()

    # svm = joblib.load("weights_save/svm_deepsbd.pkl")

    # SBD_detect(args.video, model, svm)

    # import argparse

    # parser = argparse.ArgumentParser()
    # parser.add_argument('list', help='chemin list')
    # args = parser.parse_args()

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # model = C3D_SBD().to(device)
    # model.load_state_dict(torch.load("weights_save/c3d_sbd_best_save.pth", map_location=device))
    # model.eval()
    # svm = joblib.load("weights_save/svm_deepsbd.pkl")

    # print("chargement modele ok sur ",device)

    # input_directory = r"../../../Dataset/Dataset_Shot/V3C/V3C1/videos" 
    # ech_basename = os.path.splitext(os.path.basename(args.list))[0]
    # output_directory = r"../Experimentation/DeepSBD_V3C1"  # Dossier de sortie
    # output_directory = output_directory+"_"+ech_basename

    # with open(args.list, "r", encoding="utf-8") as f:
    #     for line in f:
    #         line = line.strip()  # enlève \n et les espaces
    #         #print("Traitement de la vidéo : ",line)
    #         dir_prov=os.path.join(input_directory, line)
    #         process_videos_in_directory(dir_prov, output_directory,model,svm)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = C3D_SBD().to(device)
    model.load_state_dict(torch.load("weights_save/c3d_sbd_best.pth", map_location=device))
    model.eval()
    svm = joblib.load("weights_save/svm_deepsbd.pkl")

    # input_directory = r"../../../Dataset/Dataset_Shot/AutoShot/video" 
    # output_directory = r"../Experimentation/AutoShot_TEST/DeepSBD_AutoShot"  # Dossier de sortie
    # process_videos_in_directory(input_directory, output_directory,model,svm)

    input_directory = r"../../../Dataset/Dataset_Shot/BBC/BBC_base" 
    output_directory = r"../Experimentation/BBC_TEST/DeepSBD_bis"  # Dossier de sortie
    process_videos_in_directory(input_directory, output_directory,model,svm)


    input_directory = r"../../../Dataset/Dataset_Shot/BBC/BBC_Fade" 
    output_directory = r"../Experimentation/BBC_Fade/DeepSBD_bis"  # Dossier de sortie
    process_videos_in_directory(input_directory, output_directory,model,svm)

    input_directory = r"../../../Dataset/Dataset_Shot/BBC/BBC_FadeBlack" 
    output_directory = r"../Experimentation/BBC_FadeBlack/DeepSBD_bis"  # Dossier de sortie
    process_videos_in_directory(input_directory, output_directory,model,svm)

    input_directory = r"../../../Dataset/Dataset_Shot/BBC/BBC_FadeBlack_mirror" 
    output_directory = r"../Experimentation/BBC_FadeBlack_mirror/DeepSBD_bis"  # Dossier de sortie
    process_videos_in_directory(input_directory, output_directory,model,svm)
