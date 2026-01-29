import os
import shutil

base_path = "../../../Dataset/Dataset_Shot/V3C/V3C1/echantillons"

# Nombre total d'échantillons
nb_echantillons = 10

for i in range(1, nb_echantillons):
    #print(f"\n=== Traitement échantillon {i} ===")

    fichier_i = os.path.join(base_path, f"echantillon_{i}.txt")
    dossier_source = f"../Experimentation/DeepSBD_V3C1_echantillon_{i}"

    # Lire les lignes du fichier i
    with open(fichier_i, "r", encoding="utf-8") as f:
        lignes_i = set(ligne.strip() for ligne in f)

    # Comparer avec TOUS les fichiers suivants
    for j in range(i + 1, nb_echantillons + 1):
        #print(f"  -> Comparaison avec échantillon {j}")

        fichier_j = os.path.join(base_path, f"echantillon_{j}.txt")
        dossier_destination = f"../Experimentation/DeepSBD_V3C1_echantillon_{j}"

        os.makedirs(dossier_destination, exist_ok=True)

        # Lire les lignes du fichier j
        with open(fichier_j, "r", encoding="utf-8") as f:
            lignes_j = set(ligne.strip() for ligne in f)

        # Intersection = doublons
        doublons = lignes_i.intersection(lignes_j)

        for nom in doublons:
            fichier_a_copier = os.path.join(dossier_source, nom + ".txt")
            destination = os.path.join(dossier_destination, nom + ".txt")

            if os.path.exists(fichier_a_copier):
                shutil.copy(fichier_a_copier, destination)
                # print(f"    Copié : {nom}.txt")
