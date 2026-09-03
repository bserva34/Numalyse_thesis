import os
import time
import cv2
import torch
print(torch.version.cuda)

from ultralytics import YOLO

# Importation du nouveau module compilé en C++
from video_processor_cxx import process_video_cpp

VIDEO_DIR = "../../../../Dataset/Dataset_TagImage/Plan/"
OUT_DIR = "../../Res/Tps_ex/Yolo_m_c++/"

THR = 0.5
SIGMA_POS = 0.5
METRIC = "euclidean" # Géré nativement en C++ ici

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

        # APPEL DU CODE C++ 
        # On lui passe l'instance de notre modèle YOLO directement

        video_duration, id_frame = process_video_cpp(
            video_path,     # arg0: str
            THR,            # arg1: float
            SIGMA_POS,      # arg2: float
            80,             # arg3: int (num_classes)
            model,          # arg4: object (python_model)
            16              # arg5: int (batch_size)
        )

        # Extraction et sauvegarde de la keyframe trouvée par le C++
        if id_frame >= 0:
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(id_frame))
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                base = os.path.splitext(video_file)[0]
                out_path = os.path.join(
                    OUT_DIR,
                    f"{base}_keyframe_{id_frame:04d}.jpg"
                )
                cv2.imwrite(out_path, frame) # Décommenté pour générer l'image si besoin

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

        print(f"Durée vidéo : {video_duration:.2f}s")
        print(f"Temps exécution : {exec_time:.2f}s")
        print(f"Ratio : {ratio:.3f}")

    ratio_total = (
        total_exec_time / total_video_duration
        if total_video_duration > 0
        else 0
    )
    report.write("\n")
    report.write(f"Temps total video: {total_video_duration:.2f}s\n")
    report.write(f"Temps total : {total_exec_time:.2f}s\n")
    report.write(f"Ratio total : {ratio_total:.3f}\n")

print("\nRapport sauvegardé :")
print(report_file)