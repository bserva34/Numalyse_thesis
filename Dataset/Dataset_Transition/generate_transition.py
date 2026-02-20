import random
import subprocess
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2

# ==============================
# ARGUMENTS
# ==============================

parser = argparse.ArgumentParser()

parser.add_argument("--videos", required=True)
parser.add_argument("--gt", required=True)
parser.add_argument("--output", required=True)

parser.add_argument("--min_trans", type=int, default=30)
parser.add_argument("--max_trans", type=int, default=250)


parser.add_argument("--seed", type=int, default=42)

args = parser.parse_args()

VIDEO_EXTENSIONS = [".mp4", ".mov", ".avi", ".mkv"]

# ==============================
# UTILITAIRES
# ==============================

def read_gt_file(path):
    intervals = []
    with open(path, "r") as f:
        for line in f:
            start, end = map(int, line.strip().split())
            intervals.append((start, end))
    return intervals


def write_gt_file(path, intervals):
    with open(path, "w") as f:
        for start, end in intervals:
            f.write(f"{start} {end}\n")


def get_fps(video_path):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


# ==============================
# PROCESS VIDEO
# ==============================

def process_video(video_path, gt_path, output_video_path, output_gt_path):

    print(f"Processing {video_path.name}")

    gt_intervals = read_gt_file(gt_path)
    fps = get_fps(video_path)

    transitions = []
    new_gt = []
    filter_commands = []

    # ==============================
    # Génération transitions
    # ==============================

    for i in range(len(gt_intervals) - 1):

        start_i, end_i = gt_intervals[i]
        start_j, end_j = gt_intervals[i + 1]

        length_i = end_i - start_i 
        length_j = end_j - start_j 

        sup = min(
            args.max_trans,
            length_i // 2,
            length_j // 2
        )
        if sup > args.min_trans and length_i > 10 : 
            L = random.randint(args.min_trans,sup )
        else :
            L=0



        transitions.append(max(0, L))
    transitions.append(0)

    print(transitions)

    # ==============================
    # MODE FADE VERS NOIR
    # ==============================
    global_frame = 0

    d_in=d_out=st_out=0

    for i, (start, end) in enumerate(gt_intervals):

        print(f"Start : {start} / End : {end}")

        plan_length = end - start + 1

        L_out = transitions[i]
        L_in = transitions[i - 1] if i > 0 else 0


        start_sec = start / fps
        duration_sec = plan_length / fps

        fade_filters = []

        # Fade In
        if L_in > 0:
            d = L_in / fps
            d_in=d*fps
            fade_filters.append(f"fade=t=in:st=0:d={d}")
            print(f"st : {0*fps} / d : {d*fps}")

        # Fade Out
        if L_out > 0:
            d = L_out / fps
            d_out=d*fps
            st = (plan_length - L_out) / fps
            st_out=st*fps
            print(f"st : {st*fps} / d : {d*fps}\n")
            fade_filters.append(f"fade=t=out:st={st}:d={d}")

        fade_str = ",".join(fade_filters)

        filter_commands.append(
            f"[0:v]trim=start={start_sec}:duration={duration_sec},"
            f"setpts=PTS-STARTPTS"
            + (f",{fade_str}" if fade_str else "")
            + f"[v{i}];"
        )

        # ==============================
        # GT CORRIGÉ (zone stable)
        # ==============================

        # new_gt.append((int(start+d_in), int(start+st_out)))
        stable_start = global_frame + L_in
        stable_end = global_frame + plan_length - L_out - 1

        if stable_end >= stable_start:
            new_gt.append((stable_start, stable_end))

        global_frame += plan_length

    concat_inputs = "".join([f"[v{i}]" for i in range(len(gt_intervals))])
    filter_commands.append(
        f"{concat_inputs}concat=n={len(gt_intervals)}:v=1:a=0[outv]"
    )

    filter_complex = "".join(filter_commands)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-filter_complex", filter_complex,
        "-map", "[outv]",
        "-preset", "fast",
        str(output_video_path)
    ]

    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    write_gt_file(output_gt_path, new_gt)

    return transitions

# ==============================
# MAIN
# ==============================

def main():

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    videos_path = Path(args.videos)
    gt_path = Path(args.gt)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    video_files = [f for f in videos_path.iterdir()
                   if f.suffix.lower() in VIDEO_EXTENSIONS]

    all_transitions = []

    for video_file in video_files:

        gt_file = gt_path / (video_file.stem + ".txt")
        if not gt_file.exists():
            continue

        output_video = output_path / video_file.name
        output_gt = output_path / (video_file.stem + ".txt")

        transitions = process_video(
            video_file,
            gt_file,
            output_video,
            output_gt
        )

        all_transitions.extend(transitions)

    print("Terminé.")


if __name__ == "__main__":
    main()
