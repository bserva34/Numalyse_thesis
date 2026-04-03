import os
import numpy as np
import matplotlib.pyplot as plt

def load_and_expand(file_path, block_size=10):
    data = np.loadtxt(file_path)

    # Cas où il n'y a qu'une seule ligne
    if data.ndim == 1:
        data = np.array([data])

    # Séparer longueurs et counts
    lengths = data[:, 0].astype(int)
    counts = data[:, 1]

    # Construire un histogramme dense
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


def plot_histograms_from_folder(folder_path, normalisation=14517, block_size=20):
    plt.figure(figsize=(10, 6))

    all_data = []
    max_len = 0

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        if os.path.isfile(file_path):
            try:
                data = load_and_expand(file_path, block_size)
                all_data.append((file_name, data))
                max_len = max(max_len, len(data))
            except Exception as e:
                print(f"Erreur avec {file_name}: {e}")

    for file_name, data in all_data:
        # Padding avec 0 pour aligner toutes les courbes
        if len(data) < max_len:
            padded = np.pad(data, (0, max_len - len(data)), constant_values=0)
        else:
            padded = data

        data_percent = padded / normalisation
        x = np.arange(len(data_percent))

        plt.plot(x, data_percent, label=file_name.split("_")[0])

    plt.xlabel(f"Nombre de frames par transition")
    plt.ylabel("Pourcentage de transitions")
    plt.title("MID Metric")
    plt.legend()
    plt.grid()

    name=folder_path+"hist.png"
    plt.savefig(name)

    #plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('folder', help='folder path')
    args = parser.parse_args()

    plot_histograms_from_folder(args.folder)