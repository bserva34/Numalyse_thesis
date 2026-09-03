import os
import time
import cv2
import torch

import torchvision.models as models

from tag_image_method_faster_retina_opti import process_video


VIDEO_DIR = "../../../Dataset/Dataset_TagImage/Plan_prov/"
OUT_DIR = "../Res/Hist/Retina/"

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
print("Chargement Retina...")
print("=" * 60)

start_model = time.time()

model = models.detection.retinanet_resnet50_fpn(
    weights=models.detection.RetinaNet_ResNet50_FPN_Weights.DEFAULT
).to(device).eval()
 
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
            batch_size=8
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