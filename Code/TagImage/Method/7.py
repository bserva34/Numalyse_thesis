import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(
        description='Key frame extraction based on quaternion Fourier transform with multiple features fusion')
    parser.add_argument('video', help='Path to input video')
    parser.add_argument('--out', '-o', required=True, help='Output directory')
    parser.add_argument('--sigma', type=float, default=0.5, help='Gaussian filter sigma')
    return parser.parse_args()


def quaternion_fused_map(fm, fb, frg, fby, sigma):
    """
    Compute fused feature map via quaternion Fourier transform approximation.
    """
    f1 = fm.astype(np.float32) + 1j * fb.astype(np.float32)
    f2 = frg.astype(np.float32) + 1j * fby.astype(np.float32)

    F1 = np.fft.fft2(f1)
    F2 = np.fft.fft2(f2)

    M = np.real(F1)
    I = np.imag(F1)
    RG = np.real(F2)
    BY = np.imag(F2)

    eps = 1e-8
    phase = np.arctan(np.sqrt(RG**2 + BY**2 + I**2 + eps) / (M + eps))
    phase_norm = cv2.normalize(phase, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    ksize = int(6 * sigma + 1) | 1
    filtered = cv2.GaussianBlur(phase_norm, (ksize, ksize), sigma)

    inv = np.fft.ifft2(filtered.astype(np.float32))
    fused = np.abs(inv)
    fused_norm = cv2.normalize(fused, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return fused_norm


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    total_theoretical = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    prev = None
    fused_maps = []

    # Utilisation d'un while True pour parer à toute erreur de métadonnées de cap.get
    pbar = tqdm(total=total_theoretical, desc="Processing frames")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Correction du 'brea'
            
        b, g, r = cv2.split(frame)
        gray = ((r.astype(np.float32) + g + b) / 3).astype(np.uint8)
        fb = gray

        R = r.astype(np.float32) - (g.astype(np.float32) + b) / 2
        G = g.astype(np.float32) - (r.astype(np.float32) + b) / 2
        B = b.astype(np.float32) - (r.astype(np.float32) + g) / 2
        Y = ((r.astype(np.float32) + g) / 2) - (np.abs(r.astype(np.float32) - g) / 2)
        frg = (R - G).astype(np.uint8)
        fby = (B - Y).astype(np.uint8)

        fm = cv2.absdiff(frame, prev) if prev is not None else np.zeros_like(frame)
        prev = frame.copy()

        fused = quaternion_fused_map(fm[...,0], fb, frg, fby, args.sigma)
        fused_maps.append(fused)
        pbar.update(1)

    cap.release()
    pbar.close()

    # Nombre de frames réellement lues et valides
    actual_num_frames = len(fused_maps)

    if actual_num_frames == 0:
        print("Erreur : Aucune frame n'a pu être lue dans cette vidéo.")
        return

    # Calcul des MSE basé sur la taille réelle des cartes collectées
    mse = np.zeros(actual_num_frames)
    for i in range(1, actual_num_frames):
        diff = fused_maps[i].astype(np.float32) - fused_maps[i-1].astype(np.float32)
        mse[i] = np.mean(diff**2)

    best_idx = int(np.argmax(mse))

    # Réouverture propre pour extraire la frame sans surcharger la RAM
    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, best_idx)
    ret, best_frame = cap.read()
    cap.release()

    if ret and best_frame is not None:
        base = os.path.splitext(os.path.basename(args.video))[0]
        output_path = os.path.join(args.out, base)
        out_filename = f"{output_path}_keyframe_{best_idx:04d}.jpg"
        
        cv2.imwrite(out_filename, best_frame)
        print(f"Saved best keyframe at index: {best_idx} -> {out_filename}")
    else:
        print(f"Erreur : Impossible de lire la frame {best_idx} à la réouverture.")

if __name__ == '__main__':
    main()