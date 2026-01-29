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
    #idx = np.argmin(np.abs(mu - 1))
    omega = np.log(mu)
    idx = np.argmin(np.abs(np.real(omega)))
    return idx


# ------------------------------------------------------
# 5. PIPELINE COMPLET 
# ------------------------------------------------------

def dmd_shot_detection(video_path, window=25, threshold=150):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Impossible d'ouvrir la vidéo: {video_path}")
    except Exception as e:
        print(f"[ERREUR] {e}")
        return [],[],[]   
             
    frames = []
    cuts = []
    all_amplitudes = []
    all_delta = []

    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width, layers = frame.shape
        new_h = int(height / 6)
        new_w = int(width / 6)
        #print(f"{new_h},{new_w}")
        resize = cv2.resize(frame, (new_w, new_h))

        frames.append(resize)
        frame_id += 1

        # une fenêtre complète
        if len(frames) >= window:
            #print(frame_id)
            X1, X2 = build_snapshot_matrix(frames)
            mu, Phi = compute_dmd(X1, X2)
            alpha = compute_amplitudes(Phi, X1[:, 0])

            if len(alpha)>0 : 
                idx_bg = extract_background_mode(mu, Phi)

                amp_bg = abs(alpha[idx_bg])
                #print(amp_bg)

                if len(all_amplitudes) > 0:
                    delta = abs(amp_bg - all_amplitudes[-1])
                else:
                    delta = 0

                all_amplitudes.append(amp_bg)
                all_delta.append(delta)

                # if delta > threshold:
                #     cuts.append(frame_id)
                #     frames = []
                # else:
                #     frames[:] = frames[-1:]
                #frames[:] = frames[-4:]
                frames.pop(0)
            else : 
                all_amplitudes.append(0)
    cap.release()

    mu = float(np.mean(all_delta))
    print("mu : ",mu)
    sigma = float(np.std(all_delta))
    T = mu + 2 * sigma


    print(T)
    cpt=0
    for a in all_delta:
        if a > T : cuts.append([cpt+4,cpt+5])
        cpt+=1


    return cuts, all_amplitudes, all_delta



def Merge_res(results):
    if not results:
        return []
    merged = [results[0]]
    for r in results[1:]:
        if abs(merged[-1][1] - r[0]) <= 1 :
            merged[-1][1]=r[1]
        else:
            merged.append(r)
    return merged


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='DMD for SBD')
    parser.add_argument('video', help='chemin vers la vidéo')
    parser.add_argument('--w', type=int, default=5)
    parser.add_argument('--t', type=int, default=150)
    args = parser.parse_args()

    boundaries,amp,deltas = dmd_shot_detection(args.video,args.w,args.t)

    print("Cuts détectés :", boundaries)
    print("cuts mergés : ",Merge_res(boundaries))

    plt.figure(figsize=(14,4))
    plt.plot(amp, label='hist amplitude')
    plt.plot(deltas, label='hist amplitude')

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

    
