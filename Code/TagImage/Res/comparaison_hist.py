import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cityblock, cosine, jensenshannon

from scipy.stats import ks_2samp
from scipy.special import rel_entr

def load_all_histograms(base_dir):
    """
    Scanne automatiquement le dossier.
    Retourne :
      - data : dict[nom_methode][nom_video] = histogramme
      - all_videos : set de toutes les vidéos trouvées
    """
    data = {}
    all_videos = set()
    
    # Liste les sous-dossiers (les méthodes)
    methods = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    for method in methods:
        data[method] = {}
        # Cherche tous les fichiers .npy dans le dossier de la méthode
        search_path = os.path.join(base_dir, method, "*_avg_hist.npy")
        files = glob.glob(search_path)
        
        for f in files:
            video_name = os.path.basename(f).replace("_avg_hist.npy", "")
            all_videos.add(video_name)
            
            hist = np.load(f)
            # --- NORMALISATION IMPÉRATIVE ---
            # On s'assure que la somme de l'histogramme est égale à 1 pour pouvoir comparer
            if hist.sum() > 0:
                hist = hist / hist.sum()
            
            data[method][video_name] = hist
            
    return data, sorted(list(all_videos))

def compute_global_method_distances(data, videos, metric="cosine"):
    """
    Calcule la distance moyenne entre chaque paire de méthodes sur l'ensemble des vidéos.
    """
    methods = sorted(list(data.keys()))
    num_methods = len(methods)
    
    # Matrice pour accumuler les distances
    distance_matrix = np.zeros((num_methods, num_methods))
    # Matrice pour compter combien de vidéos ont été comparées (au cas où il y ait des manquants)
    count_matrix = np.zeros((num_methods, num_methods))
    
    for i in range(num_methods):
        for j in range(i, num_methods):
            m1 = methods[i]
            m2 = methods[j]
            
            distances = []
            for video in videos:
                if video in data[m1] and video in data[m2]:
                    h1 = data[m1][video]
                    h2 = data[m2][video]
                    
                    # Choix de la métrique
                    if metric == "cosine":
                        d = cosine(h1, h2)

                    elif metric == "Kolmogorov–Smirnov":
                        stat, pvalue = ks_2samp(h1, h2)
                        d = stat
                        # cdf1 = np.cumsum(h1)
                        # cdf2 = np.cumsum(h2)
                        # # Statistique KS : valeur maximale absolue entre les deux CDF
                        # d = np.max(np.abs(cdf1 - cdf2))

                    elif metric == "Kolmogorov–Smirnov_P-value":
                        stat, pvalue = ks_2samp(h1, h2)
                        d = pvalue                                

                    elif metric == "manhattan":
                        d = cityblock(h1, h2)

                    elif metric == "euclidean":
                        d = np.linalg.norm(h1 - h2)

                    elif metric == "jensen-shannon":
                        d = jensenshannon(h1, h2)

                    elif metric == "hellinger":
                        d = np.linalg.norm(np.sqrt(h1) - np.sqrt(h2)) / np.sqrt(2)

                    elif metric == "kullback-leibler":
                        # Pour éviter le log(0) ou division par 0, on ajoute un epsilon
                        eps = 1e-10
                        h1_safe = h1 + eps
                        h2_safe = h2 + eps
                        # Renormalisation pour rester de vraies distributions de probabilité
                        h1_safe /= h1_safe.sum()
                        h2_safe /= h2_safe.sum()
                        
                        # rel_entr(P, Q) calcule P * log(P / Q)
                        d = np.sum(rel_entr(h1_safe, h2_safe))

                    else:
                        raise ValueError(f"Métrique inconnue : {metric}")
                        
                    # On ignore les NaN (si un histogramme était complètement vide)
                    if not np.isnan(d):
                        distances.append(d)
            
            if distances:
                avg_dist = np.mean(distances)
                distance_matrix[i, j] = avg_dist
                distance_matrix[j, i] = avg_dist
                count_matrix[i, j] = len(distances)
                count_matrix[j, i] = len(distances)

    return distance_matrix, methods

if __name__ == "__main__":
    # Dossier racine où se trouvent tes sous-dossiers de méthodes
    BASE_DIRECTORY = "Hist" 
    
    # 1. Chargement et normalisation automatique
    print("Chargement des histogrammes...")
    donnees, liste_videos = load_all_histograms(BASE_DIRECTORY)
    print(f"Trouvé : {len(donnees)} méthodes et {len(liste_videos)} vidéos uniques.")
    
    metrics = [
        "cosine",
        "euclidean",
        "manhattan",
        "jensen-shannon",
        "hellinger"
    ]
    # metrics = [
    #     "Kolmogorov–Smirnov",
    #     "Kolmogorov–Smirnov_P-value",
    #     "kullback-leibler"
    # ]
    metrics = [
        "Kolmogorov–Smirnov",
        "Kolmogorov–Smirnov_P-value"
    ]

    for metric in metrics:

        matrice_dist, nom_methodes = compute_global_method_distances(
            donnees,
            liste_videos,
            metric=metric
        )

        df_dist = pd.DataFrame(
            matrice_dist,
            index=nom_methodes,
            columns=nom_methodes
        )

        df_dist.to_csv(
            os.path.join(BASE_DIRECTORY, f"distances_globales_{metric}.csv")
        )

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            df_dist,
            annot=True,
            cmap="YlGnBu",
            fmt=".4f",
            linewidths=.5,
            vmin=0.0,
            vmax=1.0
        )

        # plt.title(
        #     f"Distance moyenne entre les méthodes ({metric.capitalize()})\n"
        #     f"Calculée sur {len(liste_videos)} vidéos"
        # )

        plt.tight_layout()
        plt.savefig(
            os.path.join(BASE_DIRECTORY, f"heatmap_global_methods_{metric}.png")
        )
        plt.close()