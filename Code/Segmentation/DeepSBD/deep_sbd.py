import cv2
import torch
import numpy as np
import joblib

from models.c3d_sbd import C3D_SBD


def create_segments(video_path, length_segment=16, overlap=8):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Impossible d'ouvrir la vidéo: {video_path}")

    frames = []
    while True:
        ret, fr = cap.read()
        if not ret:
            break
        fr = cv2.resize(fr, (112, 112))
        frames.append(fr)
    cap.release()

    segments = []
    step = length_segment - overlap
    for i in range(0, len(frames) - length_segment + 1, step):
        segments.append({
            "frames": np.array(frames[i:i+length_segment]),
            "start": i,
            "end": i + length_segment - 1
        })

    return segments


def extract_c3d_features(segment_frames, model, device="cuda"):
    x = torch.from_numpy(segment_frames).float()
    x = x.permute(3,0,1,2).unsqueeze(0) / 255.0
    x = x.to(device)

    with torch.no_grad():
        feat = model(x, return_features=True)

    return feat.cpu().numpy().squeeze()


def Merge_res(results):
    if not results:
        return []
    merged = [results[0]]
    for r in results[1:]:
        if r["label"] == merged[-1]["label"]:
            merged[-1]["end"] = r["end"]
        else:
            merged.append(r)
    return merged


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
        d = bhattacharyya_distance(s["frames"][0], s["frames"][-1])
        if d < threshold:
            s["label"] = 0
    return segments


def SBD_detect(video_path, model, svm):
    segments = create_segments(video_path)
    results = []

    for s in segments:
        feat = extract_c3d_features(s["frames"], model)
        label = svm.predict([feat])[0]
        score = svm.predict_proba([feat])[0]

        results.append({
            "start": s["start"],
            "end": s["end"],
            "label": label,
            "score": score,
            "frames": s["frames"]
        })

    merged = Merge_res(results)
    final = Post_Processing(merged)
    display_res(final)


def display_res(res):
    for r in res:
        lbl = ["NO", "SHARP", "GRADUAL"][r["label"]]
        print(f"[{r['start']:6d} → {r['end']:6d}] : {lbl}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = C3D_SBD().to(device)
    model.load_state_dict(torch.load("c3d_sbd.pth", map_location=device))
    model.eval()

    svm = joblib.load("svm_deepsbd.pkl")

    SBD_detect(args.video, model, svm)
