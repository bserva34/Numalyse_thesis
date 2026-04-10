import os
import numpy as np
import matplotlib.pyplot as plt


def load_and_expand(file_path, block_size=10):
    data = np.loadtxt(file_path)

    if data.ndim == 1:
        data = np.array([data])

    lengths = data[:, 0].astype(int)
    counts = data[:, 1]

    max_len = lengths.max()
    dense = np.zeros(max_len + 1)

    for l, c in zip(lengths, counts):
        dense[l] = c

    expanded = []

    n_blocks = int(np.ceil(len(dense) / block_size))

    for i in range(n_blocks):
        start = i * block_size
        end = start + block_size
        block = dense[start:end]

        value = np.sum(block)

        expanded.extend([value] * block_size)

    return np.array(expanded)


def plot_histograms_from_folder(folder_path, norm_file, block_size=20):
    plt.figure(figsize=(10, 6))

    # 🔥 Chargement de la normalisation (par bin)
    norm_data = load_and_expand(norm_file, block_size)

    all_data = []
    max_len = len(norm_data)

    # 🔹 Chargement des fichiers
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        if os.path.isfile(file_path):
            try:
                data = load_and_expand(file_path, block_size)
                all_data.append((file_name, data))
                max_len = max(max_len, len(data))
            except Exception as e:
                print(f"Erreur avec {file_name}: {e}")

    # 🔹 Padding de la normalisation
    norm_padded = np.pad(norm_data, (0, max_len - len(norm_data)), constant_values=0)

    # 🔹 Plot
    for file_name, data in all_data:
        if len(data) < max_len:
            padded = np.pad(data, (0, max_len - len(data)), constant_values=0)
        else:
            padded = data

        # ⚠️ Division sécurisée
        with np.errstate(divide='ignore', invalid='ignore'):
            data_normalized = np.divide(padded, norm_padded)
            data_normalized[~np.isfinite(data_normalized)] = 0  # remplace inf/nan par 0

        x = np.arange(len(data_normalized))

        plt.plot(x, data_normalized*100, label=file_name.split("_")[0])

    plt.xlabel("Nombre de frames par transition")
    plt.ylabel("Ratio par bin")
    plt.title("MID Metric (normalisation par bin)")
    plt.legend()
    plt.grid()

    output_path = os.path.join(folder_path, "hist.png")
    plt.savefig(output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('folder', help='Dossier contenant les fichiers à tracer')
    parser.add_argument('norm_file', help='Fichier de normalisation')
    parser.add_argument('--block_size', type=int, default=20)

    args = parser.parse_args()

    plot_histograms_from_folder(args.folder, args.norm_file, args.block_size)