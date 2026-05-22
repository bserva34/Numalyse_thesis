import os
import numpy as np
import matplotlib.pyplot as plt


def load_data(file_path):
    data = np.loadtxt(file_path)

    if data.ndim == 1:
        data = np.array([data])

    t = data[:, 0]
    prev = data[:, 1]
    next_ = data[:, 2]
    detected = data[:, 3]

    return t, prev, next_, detected


def compute_heatmap(t, prev, next_, detected, t_bins, p_bins):
    plan = (prev + next_) / 2

    heatmap = np.zeros((len(t_bins)-1, len(p_bins)-1))
    counts = np.zeros_like(heatmap)

    for i in range(len(t)):
        t_bin = np.digitize(t[i], t_bins) - 1
        p_bin = np.digitize(plan[i], p_bins) - 1

        if 0 <= t_bin < heatmap.shape[0] and 0 <= p_bin < heatmap.shape[1]:
            heatmap[t_bin, p_bin] += detected[i]
            counts[t_bin, p_bin] += 1

    # 🔥 NORMALISATION LOCALE (celle que tu veux)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = heatmap / counts
        ratio[np.isnan(ratio)] = 0

    return ratio, counts


def plot_heatmaps(folder_path):
    # 🔹 bins
    t_bins = np.arange(0, 300, 20)
    p_bins = np.array([0, 20, 50, 100, 200, 400, 800])

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        if os.path.isfile(file_path):
            try:
                t, prev, next_, detected = load_data(file_path)

                ratio, counts = compute_heatmap(t, prev, next_, detected, t_bins, p_bins)

                ratio[counts < 3] = np.nan
                cmap = plt.cm.viridis.copy()
                cmap.set_bad(color='lightgray')

                plt.figure(figsize=(12, 8))

                im = plt.imshow(ratio.T, origin='lower', aspect='auto', vmin=0, vmax=1,cmap=cmap)
                plt.colorbar(im, label="Taux de détection")

                plt.xlabel("Longueur transition")
                plt.ylabel("Taille plan (moyenne)")

                plt.xticks(np.arange(len(t_bins)-1), t_bins[:-1])
                plt.yticks(np.arange(len(p_bins)-1), p_bins[:-1])

                plt.title(file_name.split("_")[0])

                # 🔥 AJOUT DES VALEURS DANS LES CASES
                for i in range(ratio.shape[0]):
                    for j in range(ratio.shape[1]):

                        if counts[i, j] > 2 :
                            text = f"{ratio[i,j]*100:.1f}%\n(n={int(counts[i,j])})"
                        else:
                            text = ""

                        plt.text(i, j, text,
                                 ha='center', va='center',
                                 color='white' if ratio[i,j] < 0.5 else 'black',
                                 fontsize=8)

                plt.tight_layout()

                output_path = os.path.join(folder_path, file_name + "_heatmap.png")
                plt.savefig(output_path)
                plt.close()

            except Exception as e:
                print(f"Erreur avec {file_name}: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('folder')

    args = parser.parse_args()

    plot_heatmaps(args.folder)