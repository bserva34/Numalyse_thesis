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


def build_batch_snapshots(frames_np, window, device):
    """
    frames_np : (N, P)
    return:
        X1, X2 : (B, P, window-1)
    """
    frames = torch.from_numpy(frames_np).to(device)

    # unfold temporel
    windows = frames.unfold(0, window, 1)   # (B, window, P)
    windows = windows.transpose(1, 2)       # (B, P, window)

    X1 = windows[:, :, :-1]
    X2 = windows[:, :, 1:]

    return X1, X2



def load_video_frames(video_path, stride=2):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_ids = []

    real_id = -1
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        real_id += 1
        if real_id % stride != 0:
            continue

        h, w, _ = frame.shape
        frame = cv2.resize(frame, (w // 6, h // 6))

        if is_black_frame(frame):
            continue

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ch = hsv[:, :, 0].astype(np.float32) / 255.0

        frames.append(ch.flatten())
        frame_ids.append(real_id)

    cap.release()
    return np.stack(frames), np.array(frame_ids)



# ------------------------------------------------------
# 2. DMD CLASSIQUE
# ------------------------------------------------------

def compute_dmd_batch(X1, X2, r=10):
    """
    X1, X2 : (B, P, T)
    """
    B, P, T = X1.shape

    # SVD batched
    U, S, Vh = torch.linalg.svd(X1, full_matrices=False)

    r = min(r, S.shape[-1])
    U_r = U[:, :, :r]                     # (B, P, r)
    S_r = S[:, :r]                        # (B, r)
    V_r = Vh.conj().transpose(-1, -2)[:, :, :r]  # (B, T, r)

    eps = 1e-6
    S_inv = torch.diag_embed(1.0 / torch.clamp(S_r, min=eps))

    # complex
    U_r = U_r.to(torch.complex64)
    V_r = V_r.to(torch.complex64)
    X2 = X2.to(torch.complex64)
    S_inv = S_inv.to(torch.complex64)

    # A_tilde batchée
    A_tilde = (
        U_r.conj().transpose(-1, -2)
        @ X2
        @ V_r
        @ S_inv
    )  # (B, r, r)

    # eig batched
    mu, W = torch.linalg.eig(A_tilde)

    # modes
    Phi = X2 @ V_r @ S_inv @ W  # (B, P, r)

    return mu, Phi



def compute_bg_amplitudes(mu, Phi, X1):
    """
    mu  : (B, r)
    Phi : (B, P, r)
    X1  : (B, P, T)
    """
    x1 = X1[:, :, 0].to(torch.complex64)  # (B, P)

    # pseudo-inverse batchée
    Phi_pinv = torch.linalg.pinv(Phi)
    alpha = Phi_pinv @ x1.unsqueeze(-1)   # (B, r, 1)
    alpha = alpha.squeeze(-1)             # (B, r)

    # background = |log(mu)| minimal
    omega = torch.log(torch.clamp(torch.abs(mu), min=1e-8))
    idx_bg = torch.argmin(torch.abs(torch.real(omega)), dim=1)

    amp_bg = torch.abs(alpha.gather(1, idx_bg.unsqueeze(1))).squeeze(1)
    return amp_bg



def detect_cuts_batch(amp_bg, frame_ids, stride):
    amp = amp_bg.cpu().numpy()
    delta = np.abs(np.diff(amp, prepend=amp[0]))

    T = delta.mean() + 2 * delta.std()

    cuts = []
    for i, d in enumerate(delta):
        if d > T:
            f = frame_ids[i]
            cuts.append([f - stride, f])

    return cuts, amp, delta


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
def dmd_shot_detection(video_path, window=3, r=10, stride=2, batch_size=32):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    frames_np, frame_ids = load_video_frames(video_path, stride)
    if len(frames_np) < window:
        return [], [], []

    X1, X2 = build_batch_snapshots(frames_np, window, device)

    B = X1.shape[0]

    amp_all = []

    for i in range(0, B, batch_size):
        X1_b = X1[i:i+batch_size]
        X2_b = X2[i:i+batch_size]

        #a tester
        # with torch.cuda.amp.autocast(dtype=torch.float16):
        #     mu, Phi = compute_dmd_batch(X1_b, X2_b, r)

        with torch.no_grad():  
            mu, Phi = compute_dmd_batch(X1_b, X2_b, r)
            amp_bg = compute_bg_amplitudes(mu, Phi, X1_b)

        amp_all.append(amp_bg)

        del mu, Phi, amp_bg
        torch.cuda.empty_cache()

    amp_bg = torch.cat(amp_all, dim=0)

    return detect_cuts_batch(
        amp_bg,
        frame_ids[window-1:],
        stride
    )





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
    import argparse
    parser = argparse.ArgumentParser(description='DMD for SBD')
    parser.add_argument('video', help='chemin vers la vidéo')
    parser.add_argument('--w', type=int, default=5)
    args = parser.parse_args()

    boundaries,amp,deltas = dmd_shot_detection(args.video,args.w)

    print("Cuts détectés :", boundaries)
    print("cuts mergés : ",Merge_res(boundaries))

    plt.figure(figsize=(14,4))
    plt.plot(amp, label='hist amplitude')
    plt.plot(deltas, label='hist amplitude')

    plt.show()

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




    
