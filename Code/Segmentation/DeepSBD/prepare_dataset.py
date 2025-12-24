import os
import cv2
import lmdb
import pickle
import argparse
import numpy as np
from tqdm import tqdm

LABELS = {
    "no": 0,
    "sharp": 1,
    "gradual": 2
}

def load_gt_csv(path):
    """
    CSV normalisé :
    video,start,end,label
    """
    gt = {}
    with open(path, "r") as f:
        next(f)
        for line in f:
            v, s, e, lbl = line.strip().split(",")
            s, e = int(s), int(e)
            if v not in gt:
                gt[v] = []
            gt[v].append((s, e, LABELS[lbl]))
    return gt


def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, fr = cap.read()
        if not ret:
            break
        fr = cv2.resize(fr, (112, 112))
        fr = fr[..., ::-1]  # BGR -> RGB
        frames.append(fr)
    cap.release()
    return frames


def extract_segments(frames, gt_list, seg_len=16, overlap=8):
    step = seg_len - overlap
    segments = []

    for start in range(0, len(frames) - seg_len + 1, step):
        end = start + seg_len - 1
        label = 0  # no-transition

        for s, e, l in gt_list:
            if s <= end and e >= start:
                label = l
                break

        clip = frames[start:start+seg_len]
        clip = np.stack(clip)            # (T,H,W,C)
        clip = clip.transpose(3,0,1,2)   # (C,T,H,W)
        clip = clip.astype(np.float32) / 255.0

        segments.append((clip, label))

    return segments

def estimate_map_size(video_dir, gt, seg_len=16, overlap=8):
    total_segments = 0

    for vid in os.listdir(video_dir):
        if not vid.endswith(".mp4"):
            continue
        if vid not in gt:
            continue

        cap = cv2.VideoCapture(os.path.join(video_dir, vid))
        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        step = seg_len - overlap
        nseg = max(0, (nframes - seg_len) // step)
        total_segments += nseg

    bytes_per_segment = 2.6 * 1024**2  # 2.6 MB
    total_bytes = int(total_segments * bytes_per_segment * 1.2)  # +20%

    print(f"[INFO] Segments estimés : {total_segments}")
    print(f"[INFO] map_size ≈ {total_bytes / 1024**3:.1f} Go")

    return total_bytes



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", required=True)
    parser.add_argument("--gt", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    gt = load_gt_csv(args.gt)

    map_size = estimate_map_size(args.videos, gt)

    env = lmdb.open(
        args.out,
        map_size=map_size,  
        map_async=True
    )

    idx = 0
    with env.begin(write=True) as txn:
        for vid in tqdm(os.listdir(args.videos)):
            if not vid.endswith(".mp4"):
                continue
            if vid not in gt:
                continue

            print(f"[INFO] {vid}")
            frames = read_video(os.path.join(args.videos, vid))
            segments = extract_segments(frames, gt[vid])

            for clip, label in segments:
                key = f"{idx:010d}".encode("ascii")
                value = pickle.dumps({
                    "video": clip,
                    "label": label
                })
                txn.put(key, value)
                idx += 1

        txn.put(b"__len__", pickle.dumps(idx))

    env.sync()
    env.close()
    print(f"[DONE] {idx} segments écrits dans LMDB")


if __name__ == "__main__":
    main()
