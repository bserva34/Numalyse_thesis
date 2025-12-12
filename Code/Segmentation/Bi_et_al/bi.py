import cv2
import numpy as np
from numpy.linalg import svd, eig, pinv
import matplotlib.pyplot as plt


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
        ch /= 255.0  # normalisation
        flat.append(ch.flatten())
    X = np.array(flat).T   # shape (pixels, time)
    return X[:, :-1], X[:, 1:]



# ------------------------------------------------------
# 2. DMD CLASSIQUE
# ------------------------------------------------------

def compute_dmd(X1, X2, r=None, svd_thresh=1e-10):

    U, S, Vh = svd(X1, full_matrices=False)

    # Filtrage automatique
    if r is None:
        r = np.sum(S > svd_thresh)

    U_r = U[:, :r]
    S_r = S[:r]
    V_r = Vh.conj().T[:, :r]

    A_tilde = U_r.conj().T @ X2 @ V_r @ np.diag(1 / S_r)
    mu, W = eig(A_tilde)

    Phi = X2 @ V_r @ np.diag(1 / S_r) @ W

    return mu, Phi



def compute_amplitudes(Phi, x1):
    return pinv(Phi) @ x1


# ------------------------------------------------------
# 3. FONCTIONS POUR BACKGROUND
# ------------------------------------------------------

def extract_background_mode(mu, Phi):
    # background = mode le plus proche de mu = 1
    idx = np.argmin(np.abs(mu - 1))
    return idx


# ------------------------------------------------------
# 5. PIPELINE COMPLET 
# ------------------------------------------------------

def dmd_shot_detection(video_path, window=25, threshold=150):

    cap = cv2.VideoCapture(video_path)
    frames = []
    cuts = []
    all_amplitudes = []

    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frames.append(frame)
        frame_id += 1

        # une fenêtre complète
        if len(frames) >= window:
            print(frame_id)
            X1, X2 = build_snapshot_matrix(frames)
            mu, Phi = compute_dmd(X1, X2)
            alpha = compute_amplitudes(Phi, X1[:, 0])

            if len(alpha)>0 : 
                idx_bg = extract_background_mode(mu, Phi)

                amp_bg = abs(alpha[idx_bg])
                print(amp_bg)
                all_amplitudes.append(amp_bg)

                if amp_bg > threshold:
                    # coupure = dernière frame de la fenêtre
                    cuts.append(frame_id)
                    frames = []  # recommence après le cut
                else:
                    #frames[:] = frames[-1:]
                    frames.pop(0)
            else : 
                all_amplitudes.append(0)

    cap.release()
    return cuts, all_amplitudes



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='DMD for SBD')
    parser.add_argument('video', help='chemin vers la vidéo')
    parser.add_argument('--w', type=int, default=25)
    parser.add_argument('--t', type=int, default=150)
    args = parser.parse_args()

    boundaries,amp = dmd_shot_detection(args.video,args.w,args.t)

    print("Cuts détectés :", boundaries)

    plt.figure(figsize=(14,4))
    plt.plot(amp, label='hist amplitude')

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
    plt.show()

    
