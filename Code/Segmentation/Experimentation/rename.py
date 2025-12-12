import os

dossier = "AdaptiveDetector_ClipShots"  # changer si nécessaire

for nom in os.listdir(dossier):
    if nom.endswith("_AdaptiveDetector.txt"):
        nouveau_nom = nom.replace("_AdaptiveDetector", "")
        os.rename(
            os.path.join(dossier, nom),
            os.path.join(dossier, nouveau_nom)
        )
        print(f"Renommé : {nom} → {nouveau_nom}")
