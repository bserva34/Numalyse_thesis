import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="Folder containing prediction files")
    args = parser.parse_args()


# dossier contenant les fichiers
dossier = args.folder

plt.figure(figsize=(10, 6))

for fichier in os.listdir(dossier):
    if fichier.endswith(".txt"):
        chemin = os.path.join(dossier, fichier)
        data = np.loadtxt(chemin)
        y = data[:, 1]
        x = np.arange(1, len(y) + 1)

        plt.plot(x, y, label=fichier, linewidth=0.7)



#différence par rapport au premier 
# reference = None

# for fichier in os.listdir(dossier):
#     if fichier.endswith(".txt"):
#         chemin = os.path.join(dossier, fichier)

#         # lire les données
#         data = np.loadtxt(chemin)

#         # deuxième colonne
#         y = data[:, 1]

#         if reference is None:
#             reference = y

#         diff = y - reference
#         x = np.arange(1,len(y)+1)

#         plt.plot(x, diff, label=fichier)


plt.xlabel("Indice de Frame")
plt.ylabel("Prédiction")
plt.title("Comparaison TransNetV2")
plt.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, label='Seuil 0.5')

#vv dissolve
#plt.axvspan(100, 200, facecolor='blue', alpha=0.2, edgecolor='none', label='VV')

#vv dissolve2
#plt.axvspan(175, 225, facecolor='blue', alpha=0.2, edgecolor='none', label='VV')

#vv dissolve3
#plt.axvspan(825, 900, facecolor='blue', alpha=0.2, edgecolor='none', label='VV')

#vv wipe
# plt.axvspan(0, 12, facecolor='blue', alpha=0.2, edgecolor='none', label='VV')
# plt.axvspan(29, 33, facecolor='blue', alpha=0.2, edgecolor='none')
# plt.axvspan(90, 94, facecolor='blue', alpha=0.2, edgecolor='none')
# plt.axvspan(106, 150, facecolor='blue', alpha=0.2, edgecolor='none')

#vv mc
# plt.axvspan(296, 301, facecolor='blue', alpha=0.2, edgecolor='none', label='VV')
# plt.axvspan(544, 549, facecolor='blue', alpha=0.2, edgecolor='none')
# plt.axvspan(599, 604, facecolor='blue', alpha=0.2, edgecolor='none')
# plt.axvspan(677, 682, facecolor='blue', alpha=0.2, edgecolor='none')
# plt.axvspan(851, 856, facecolor='blue', alpha=0.2, edgecolor='none')
# plt.axvspan(1028, 1033, facecolor='blue', alpha=0.2, edgecolor='none')
# plt.axvspan(1170, 1175, facecolor='blue', alpha=0.2, edgecolor='none')

#vv shot dissolve
# plt.axvspan(0, 14, facecolor='blue', alpha=0.2, edgecolor='none', label='VV')
# plt.axvspan(111, 130, facecolor='blue', alpha=0.2, edgecolor='none')
# plt.axvspan(244, 260, facecolor='blue', alpha=0.2, edgecolor='none')

plt.legend()
plt.tight_layout()
name_save = dossier+"comparaison.png"
plt.savefig(name_save,dpi=400)
# plt.show()
plt.close()