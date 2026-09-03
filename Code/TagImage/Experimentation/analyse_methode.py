import argparse
import os
import statistics
import matplotlib.pyplot as plt
import numpy as np
from numpy import trapz


def load_distances(filepath):
    distances = []
    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            _, dist = parts
            distances.append(float(dist))

    if not distances:
        return [], float('nan'), float('nan'), float('nan'), float('nan'), float('nan')

    mean = statistics.mean(distances)
    median = statistics.median(distances)
    sd = statistics.pstdev(distances)
    q = statistics.quantiles(distances, n=4)
    iqr = q[2] - q[0]
    maximum = max(distances)

    return distances, mean, median, sd, iqr, maximum


def precision_for_thresholds(distances, thresholds):
    precisions = []
    n = len(distances)

    for t in thresholds:
        tp = sum(d <= t for d in distances)
        fp = n - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        precisions.append(precision)

    return precisions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", help="Dossier contenant les fichiers .txt")
    parser.add_argument("--step", type=float, default=0.05,
                        help="Incrément du seuil (default=0.05)")
    args = parser.parse_args()

    # Nettoyage du chemin pour éviter les problèmes avec les slashes de fin
    target_dir = os.path.abspath(args.results_dir)
    folder_name = os.path.basename(target_dir)
    parent_dir = os.path.dirname(target_dir)

    # Chemin du fichier texte dans le dossier parent
    prov = f"{target_dir}.txt"
    log_filepath = os.path.join(parent_dir, prov)

    thresholds = np.arange(0.0, 1.0 + args.step, args.step)

    plt.figure()
    plt.xlabel("Threshold")
    plt.ylabel("Acceptance Rate")
    plt.grid(True)

    # Ouverture du fichier en mode écriture ('w')
    with open(log_filepath, "w", encoding="utf-8") as log_file:
        
        for fname in os.listdir(target_dir):
            if not fname.endswith(".txt"):
                continue

            path = os.path.join(target_dir, fname)
            distances, mean, median, sd, et, maximum = load_distances(path)
            
            if not distances:
                continue

            precisions = precision_for_thresholds(distances, thresholds)
            auc = trapz(precisions, thresholds)

            # Préparation du texte à afficher ET à écrire
            output = (
                f"Méthode : {fname}\n"
                f"Moyenne : {mean}\n"
                f"Ecart type : {sd}\n"
                f"Médiane : {median}\n"
                f"Ecart inter-quartile : {et}\n"
                f"Maximum : {maximum}\n"
                f"Aire sous la courbe (AUC) : {auc}\n"
                f"---------\n"
            )
            
            # Affichage console + Ériture fichier
            print(output, end="")
            log_file.write(output)

            label = os.path.splitext(fname)[0]
            plt.plot(thresholds, precisions, label=label)

    plt.legend(fontsize=10)
    
    # Sauvegarde de la figure avec le nom du dossier traité
    figure_name = f"{parent_dir}.png"

    prov = f"{target_dir}.png"
    figure_name = os.path.join(parent_dir, prov)

    plt.savefig(figure_name, dpi=800)
    print(f"\n[OK] Graphique sauvegardé sous : {figure_name}")
    print(f"[OK] Stats sauvegardées dans : {log_filepath}")