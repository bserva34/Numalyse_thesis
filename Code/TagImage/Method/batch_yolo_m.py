import os
import time
import cv2
import torch
print(torch.version.cuda)

from ultralytics import YOLO

from tag_image_method_rtdetr_yolo_opti import process_video


VIDEO_DIR = "../../../../../benjamin-serva/Extreme SSD/Film_Numalyse/Dataset_Plan_Q2_corrige/video_corrige/"
#VIDEO_DIR = "../../../Dataset/Dataset_TagImage/Plan_ss_ensemble/"
OUT_DIR = "../Res/Q2_corrige/Yolom/"

THR = 0.5
SIGMA_POS = 0.5
METRIC = "euclidean"

os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("=" * 60)
print("Device :", device)
print("Chargement Yolo...")
print("=" * 60)

start_model = time.time()

model = YOLO("yolov8m.pt")
#model = YOLO("yolov8l.pt")
 
print(
    f"Modèle chargé en "
    f"{time.time()-start_model:.2f}s"
)

video_files = sorted([
    f for f in os.listdir(VIDEO_DIR)
    if f.lower().endswith(
        (
            ".mp4",
            ".avi",
            ".mov",
            ".mkv"
        )
    )
])

report_file = os.path.join(
    OUT_DIR,
    "execution_times.txt"
)

total_exec_time = 0
total_video_duration = 0

with open(report_file, "w") as report:

    report.write(
        "video\tvideo_duration(s)\texecution_time(s)\tratio\n"
    )

    for video_file in video_files:

        video_path = os.path.join(
            VIDEO_DIR,
            video_file
        )

        print("\n" + "=" * 60)
        print(video_file)
        print("=" * 60)

        start = time.time()

        video_duration = process_video(
            model=model,
            video_path=video_path,
            out_dir=OUT_DIR,
            thr=THR,
            metric=METRIC,
            sigma_pos=SIGMA_POS,
            batch_size=16
        )

        exec_time = time.time() - start

        total_exec_time += exec_time
        total_video_duration += video_duration

        ratio = (
            exec_time / video_duration
            if video_duration > 0
            else 0
        )

        report.write(
            f"{video_file}\t"
            f"{video_duration:.2f}\t"
            f"{exec_time:.2f}\t"
            f"{ratio:.3f}\n"
        )

        print(
            f"Durée vidéo : {video_duration:.2f}s"
        )

        print(
            f"Temps exécution : {exec_time:.2f}s"
        )

        print(
            f"Ratio : {ratio:.3f}"
        )

    ratio_total = (
    total_exec_time / total_video_duration
    if total_video_duration > 0
    else 0
    )
    report.write("\n")
    report.write(
        f"Temps total video: "
        f"{total_video_duration:.2f}s\n"
    )
    report.write(
        f"Temps total : "
        f"{total_exec_time:.2f}s\n"
    )
    report.write(
        f"Ratio total : "
        f"{ratio_total:.3f}\n"
    )

print("\nRapport sauvegardé :")
print(report_file)