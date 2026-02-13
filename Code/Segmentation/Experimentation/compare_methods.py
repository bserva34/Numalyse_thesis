import os
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import argparse


# ===========================
# Lecture des fichiers résultats
# ===========================
def parse_result_file(filepath):
    """
    Extrait les métriques globales STRICT et COOL
    """
    metrics = {}

    with open(filepath, "r") as f:
        lines = f.readlines()

    for line in lines:
        if line.startswith("Precision |"):
            parts = re.findall(r"mean=([0-9.]+)", line)
            metrics["Precision"] = float(parts[0])

        if line.startswith("Recall |"):
            parts = re.findall(r"mean=([0-9.]+)", line)
            metrics["Recall"] = float(parts[0])

        if line.startswith("F1 |"):
            parts = re.findall(r"mean=([0-9.]+)", line)
            metrics["F1"] = float(parts[0])

        if line.startswith("Precision_cool"):
            parts = re.findall(r"mean=([0-9.]+)", line)
            metrics["Precision_cool"] = float(parts[0])

        if line.startswith("Recall_cool"):
            parts = re.findall(r"mean=([0-9.]+)", line)
            metrics["Recall_cool"] = float(parts[0])

        if line.startswith("F1_cool"):
            parts = re.findall(r"mean=([0-9.]+)", line)
            metrics["F1_cool"] = float(parts[0])

    return metrics


# ===========================
# MAIN
# ===========================
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("results_folder", help="Folder containing result files")
    args = parser.parse_args()

    data = defaultdict(lambda: defaultdict(dict))

    # ===========================
    # Charger tous les fichiers
    # ===========================
    for filename in os.listdir(args.results_folder):

        if not filename.endswith(".txt"):
            continue

        match = re.match(r"(.*)_results_tol_(\d+)\.txt", filename)
        if not match:
            continue

        method = match.group(1)
        tol = int(match.group(2))

        filepath = os.path.join(args.results_folder, filename)
        metrics = parse_result_file(filepath)

        data[method][tol] = metrics

    # ===========================
    # Courbes F1 vs tolérance
    # ===========================
    plt.figure()

    for method in data:

        tolerances = sorted(data[method].keys())
        f1_scores = [data[method][t]["F1"] for t in tolerances]

        plt.plot(tolerances, f1_scores, marker='o', label=method)

    plt.xlabel("Tolérance (frames)")
    plt.ylabel("F1 score (STRICT)")
    plt.title("F1 vs Tolérance")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.dirname(args.results_folder)+"/comparison_F1_strict.png")
    plt.close()


    # ===========================
    # Courbes F1 COOL
    # ===========================
    plt.figure()

    for method in data:

        tolerances = sorted(data[method].keys())
        f1_scores = [data[method][t]["F1_cool"] for t in tolerances]

        plt.plot(tolerances, f1_scores, marker='o', label=method)

    plt.xlabel("Tolérance (frames)")
    plt.ylabel("F1 score (COOL)")
    plt.title("F1 COOL vs Tolérance")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.dirname(args.results_folder)+"/comparison_F1_cool.png")
    plt.close()


    print("✅ Graphiques générés :")
    print(" - comparison_F1_strict.png")
    print(" - comparison_F1_cool.png")


