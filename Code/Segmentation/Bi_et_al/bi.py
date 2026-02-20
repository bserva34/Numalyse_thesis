import cv2
import numpy as np
from numpy.linalg import svd, eig, pinv
import matplotlib.pyplot as plt
import os
import torch


# ------------------------------------------------------
# 1. UTILITAIRES COULEURS & MATRICES DMD
# ------------------------------------------------------

def rgb_to_hsv_channel(frame, channel="h"):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return hsv[:, :, 0] if channel == "h" else hsv[:, :, 1]


def build_snapshot_matrix(frames, channel="h"):
    flat = []
    for f in frames:
        ch = rgb_to_hsv_channel(f, channel).astype(np.float32)
        ch /= 255.0
        flat.append(ch.flatten())
    X = np.stack(flat, axis=1)  # (pixels, time)
    return X[:, :-1], X[:, 1:]




# ------------------------------------------------------
# 2. DMD CLASSIQUE
# ------------------------------------------------------

def compute_dmd_torch(X1_np, X2_np, r=10, device="cuda"):
    # --- CPU -> GPU
    X1 = torch.as_tensor(X1_np, dtype=torch.float32, device=device)
    X2 = torch.as_tensor(X2_np, dtype=torch.float32, device=device)

    # --- SVD
    U, S, Vh = torch.linalg.svd(X1, full_matrices=False)

    r = min(r, S.numel())
    U_r = U[:, :r]
    S_r = S[:r]
    V_r = Vh.conj().T[:, :r]

    # --- Sécurité numérique
    eps = 1e-6
    S_r_safe = torch.clamp(S_r, min=eps)

    # --- Passage en complex
    U_r = U_r.to(torch.complex64)
    V_r = V_r.to(torch.complex64)
    X2 = X2.to(torch.complex64)
    S_r_safe = S_r_safe.to(torch.complex64)

    # --- A_tilde
    A_tilde = (
        U_r.conj().T
        @ X2
        @ V_r
        @ torch.diag(1.0 / S_r_safe)
    )

    # --- Eigendecomposition
    mu, W = torch.linalg.eig(A_tilde)

    # --- Modes DMD
    Phi = X2 @ V_r @ torch.diag(1.0 / S_r_safe) @ W

    return mu.cpu().numpy(), Phi.cpu().numpy()





def compute_amplitudes_torch(Phi_np, x1_np, device="cuda"):
    Phi = torch.as_tensor(Phi_np, dtype=torch.complex64, device=device)
    x1 = torch.as_tensor(x1_np, dtype=torch.complex64, device=device)

    if torch.norm(x1) < 1e-6:
        return np.zeros(Phi.shape[1])

    alpha = torch.linalg.pinv(Phi) @ x1
    return alpha.cpu().numpy()




# ------------------------------------------------------
# 3. FONCTIONS POUR BACKGROUND
# ------------------------------------------------------

def extract_background_mode(mu):
    eps = 1e-8
    omega = np.log(np.maximum(np.abs(mu), eps))
    return np.argmin(np.abs(np.real(omega)))



def is_black_frame(frame, thresh=5):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return np.mean(gray) < thresh


# ------------------------------------------------------
# 5. PIPELINE COMPLET 
# ------------------------------------------------------
# 1 frame sur 2
# def dmd_shot_detection(video_path, window=3, r=10, stride=2):
#     """
#     DMD shot boundary detection with temporal subsampling.
#     stride=2 => one frame out of two
#     """

#     device = "cuda" if torch.cuda.is_available() else "cpu"


#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print(f"[ERREUR] Impossible d'ouvrir {video_path}")
#         return [], [], []

#     frames = []
#     frame_ids = []   # indices réels dans la vidéo

#     all_amplitudes = []
#     all_delta = []
#     dmd_frame_ids = []
#     cuts = []

#     real_frame_id = -1

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         real_frame_id += 1

#         if real_frame_id % stride != 0:
#             continue

#         h, w, _ = frame.shape
#         frame = cv2.resize(frame, (w // 6, h // 6))

#         # frames.append(frame)
#         # frame_ids.append(real_frame_id)
#         if is_black_frame(frame):
#             continue  

#         frames.append(frame)
#         frame_ids.append(real_frame_id)


#         if len(frames) >= window:
#             # --- Snapshot matrix
#             X1, X2 = build_snapshot_matrix(frames)

#             # --- DMD
#             mu, Phi = compute_dmd_torch(X1, X2, r=r, device=device)
#             alpha = compute_amplitudes_torch(Phi, X1[:, 0], device=device)

#             if len(alpha) > 0:
#                 idx_bg = extract_background_mode(mu)
#                 amp_bg = np.abs(alpha[idx_bg])

#                 if all_amplitudes:
#                     delta = abs(amp_bg - all_amplitudes[-1])
#                 else:
#                     delta = 0.0

#                 all_amplitudes.append(amp_bg)
#                 all_delta.append(delta)

#                 # frame réelle associée à cette DMD
#                 dmd_frame_ids.append(frame_ids[-1])

#             # --- Fenêtre glissante
#             frames.pop(0)
#             frame_ids.pop(0)

#     cap.release()

#     # --------------------------------------------------
#     # Seuil adaptatif (stable car stats cohérentes)
#     # --------------------------------------------------
#     if all_delta:
#         mu_d = float(np.mean(all_delta))
#         sigma_d = float(np.std(all_delta))
#         T = mu_d + 2 * sigma_d
#     else:
#         T = 0.0

#     # --------------------------------------------------
#     # Détection des cuts (indices réels)
#     # --------------------------------------------------
#     for i, d in enumerate(all_delta):
#         if d > T:
#             f = dmd_frame_ids[i]
#             cuts.append([f - stride, f])

#     return cuts, all_amplitudes, all_delta



def dmd_shot_detection(video_path, window=5, r=10):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERREUR] Impossible d'ouvrir {video_path}")
        return [], [], []

    frames = []
    cuts = []
    all_amplitudes = []
    all_delta = []
    dmd_frame_ids = []

    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        frame = cv2.resize(frame, (w // 6, h // 6))

        frame_id += 1

        if is_black_frame(frame):
            continue  

        frames.append(frame)
        

        if len(frames) >= window:
            X1, X2 = build_snapshot_matrix(frames)
            mu, Phi = compute_dmd_torch(X1, X2, r=r, device=device)
            alpha = compute_amplitudes_torch(Phi, X1[:, 0], device=device)

            if len(alpha) > 0:
                idx_bg = extract_background_mode(mu)
                amp_bg = np.abs(alpha[idx_bg])

                delta = (
                    abs(amp_bg - all_amplitudes[-1])
                    if all_amplitudes else 0
                )

                all_amplitudes.append(amp_bg)
                all_delta.append(delta)
                dmd_frame_ids.append(frame_id)

            frames.pop(0)

    cap.release()

    # # ---- seuil adaptatif
    mu_d = float(np.mean(all_delta)) if all_delta else 0
    sigma_d = float(np.std(all_delta)) if all_delta else 0
    T_GLOBAL = mu_d + 2. * sigma_d
    print("Seuil global : ",T_GLOBAL)


    for i, d in enumerate(all_delta):
        if d > T_GLOBAL:
            f = dmd_frame_ids[i]
            cuts.append([f - 1, f])

    # window_stat = 250

    # for i in range(len(all_delta)):
    #     start = max(0, i - window_stat // 2)
    #     end = min(len(all_delta), i + window_stat // 2)

    #     local = all_delta[start:end]

    #     mu = np.mean(local)
    #     sigma = np.std(local)
    #     T = mu + 5 * sigma

    #     if all_delta[i] > T :
    #         f = dmd_frame_ids[i]
    #         cuts.append([f - 1, f])
    #         #print(T)


    return cuts, all_amplitudes, all_delta



def Merge_res(results):
    if not results:
        return []
    merged = [results[0]]
    for r in results[1:]:
        if abs(merged[-1][1] - r[0]) <= 1:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return merged


def write_res(output_file, res):
    with open(output_file, "w") as f:
        for r in res:
            f.write(f"{r[0]}\t{r[1]}\n")


def process_videos_in_directory(directory, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    video_files = [
        f for f in os.listdir(directory)
        if f.lower().endswith(('.mp4', '.avi', '.mkv', '.mov'))
    ]

    for video in video_files:
        video_path = os.path.join(directory, video)
        name = os.path.splitext(video)[0]
        out = os.path.join(output_dir, f"{name}.txt")

        if os.path.exists(out):
            continue

        print("Traitement :", name)
        res, _, _ = dmd_shot_detection(video_path)
        write_res(out, Merge_res(res))

if __name__ == '__main__':
    # import argparse
    # parser = argparse.ArgumentParser(description='DMD for SBD')
    # parser.add_argument('video', help='chemin vers la vidéo')
    # parser.add_argument('--w', type=int, default=5)
    # args = parser.parse_args()

    # boundaries,amp,deltas = dmd_shot_detection(args.video,args.w)

    # print("Cuts détectés :", boundaries)
    # print("cuts mergés : ",Merge_res(boundaries))

    # plt.figure(figsize=(14,4))
    # plt.plot(amp, label='hist amplitude')
    # plt.plot(deltas, label='hist amplitude')

    # plt.show()

    # import argparse

    # parser = argparse.ArgumentParser()
    # parser.add_argument('list', help='chemin list')
    # args = parser.parse_args()


    # input_directory = r"../../../Dataset/Dataset_Shot/V3C/V3C1/videos" 
    # ech_basename = os.path.splitext(os.path.basename(args.list))[0]
    # output_directory = r"../Experimentation/Bi_V3C1"  # Dossier de sortie
    # output_directory = output_directory+"_"+ech_basename

    # with open(args.list, "r", encoding="utf-8") as f:
    #     for line in f:
    #         line = line.strip()  # enlève \n et les espaces
    #         #print("Traitement de la vidéo : ",line)
    #         dir_prov=os.path.join(input_directory, line)
    #         process_videos_in_directory(dir_prov, output_directory)


    #long dissolve transition
    # plt.axvspan(100-args.w, 180-args.w, alpha=0.12, color='red')
    # plt.text(100-args.w, max(amp), "vv", rotation=90, fontsize=8)

    #adele cut
    # plt.axvspan(38-args.w, 39-args.w, alpha=0.12, color='red')
    # plt.text(38-args.w, max(amp), "vv", rotation=90, fontsize=8)

    #adele multi cut
    # plt.axvspan(298-args.w, 299-args.w, alpha=0.12, color='red')
    # plt.axvspan(546-args.w, 547-args.w, alpha=0.12, color='red')
    # plt.axvspan(601-args.w, 602-args.w, alpha=0.12, color='red')
    # plt.axvspan(679-args.w, 680-args.w, alpha=0.12, color='red')
    # plt.axvspan(853-args.w, 854-args.w, alpha=0.12, color='red')
    # plt.axvspan(1030-args.w, 1031-args.w, alpha=0.12, color='red')
    # plt.axvspan(1172-args.w, 1173-args.w, alpha=0.12, color='red')

    # plt.text(298-args.w, max(amp), "vv", rotation=90, fontsize=8)
    # plt.text(546-args.w, max(amp), "vv", rotation=90, fontsize=8)
    # plt.text(601-args.w, max(amp), "vv", rotation=90, fontsize=8)
    # plt.text(679-args.w, max(amp), "vv", rotation=90, fontsize=8)
    # plt.text(853-args.w, max(amp), "vv", rotation=90, fontsize=8)      
    # plt.text(1030-args.w, max(amp), "vv", rotation=90, fontsize=8)
    # plt.text(1172-args.w, max(amp), "vv", rotation=90, fontsize=8)
    
    # input_directory = r"../../../Dataset/Dataset_Shot/BBC/videos" 
    # output_directory = r"../Experimentation/BBC_TEST/Bi_bis_fenetre_BBC"  # Dossier de sortie

    # input_directory = r"../../../Dataset/Dataset_Shot/AutoShot/video" 
    # output_directory = r"../Experimentation/AutoShot_TEST/Bi_fenetre_AutoShot"  # Dossier de sortie

    # input_directory = r"../../../Dataset/Dataset_Shot/ClipShots/videos" 
    # output_directory = r"../Experimentation/ClipShots_TEST/Bi_ClipShots"  # Dossier de sortie
    
    #process_videos_in_directory(input_directory, output_directory)

    input_directory = r"../../../Dataset/Dataset_Shot/BBC/BBC_Fade" 
    output_directory = r"../Experimentation/BBC_Fade/Bi_2"  # Dossier de sortie
    process_videos_in_directory(input_directory, output_directory)

    input_directory = r"../../../Dataset/Dataset_Shot/BBC/BBC_FadeBlack" 
    output_directory = r"../Experimentation/BBC_FadeBlack/Bi_2"  # Dossier de sortie
    process_videos_in_directory(input_directory, output_directory)

    input_directory = r"../../../Dataset/Dataset_Shot/BBC/BBC_FadeBlack_mirror" 
    output_directory = r"../Experimentation/BBC_FadeBlack_mirror/Bi_2"  # Dossier de sortie
    process_videos_in_directory(input_directory, output_directory)


    
