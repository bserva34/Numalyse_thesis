import csv
import torch
import torchvision.models as models
from ultralytics import YOLO, RTDETR

# 1. Configuration du device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Récupération des labels pour Faster R-CNN et RetinaNet (via torchvision)
weights_frcnn = models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
weights_retina = models.detection.RetinaNet_ResNet50_FPN_Weights.DEFAULT

labels_frcnn = weights_frcnn.meta["categories"]
labels_retina = weights_retina.meta["categories"]

# 3. Récupération des labels pour YOLOv8 et RT-DETR (via ultralytics)
model_yolo = YOLO("yolov8m.pt")
model_rtdetr = RTDETR("rtdetr-l.pt")

labels_yolo = model_yolo.names
labels_rtdetr = model_rtdetr.names

# 4. Alignement et écriture dans le fichier CSV
max_length = max(len(labels_frcnn), len(labels_retina), len(labels_yolo), len(labels_rtdetr))
csv_filename = "comparatif_labels_models.csv"

print(f"Génération du fichier {csv_filename} en cours...")

# Ouverture du fichier en mode écriture ('w') avec encodage UTF-8
with open(csv_filename, mode='w', newline='', encoding='utf-8') as f:
    # Utilisation du point-virgule comme délimiteur pour une compatibilité native avec Excel (Europe)
    writer = csv.writer(f, delimiter=';')
    
    # Écriture de l'en-tête
    writer.writerow(["ID", "Faster R-CNN", "RetinaNet", "YOLOv8-M", "RT-DETR-L"])
    
    # Écriture des lignes de données
    for i in range(max_length):
        lbl_frcnn  = labels_frcnn[i] if i < len(labels_frcnn) else "-"
        lbl_retina = labels_retina[i] if i < len(labels_retina) else "-"
        
        # Les dictionnaires d'Ultralytics utilisent des clés entières {0: "person", ...}
        lbl_yolo   = labels_yolo[i] if i in labels_yolo else "-"
        lbl_rtdetr = labels_rtdetr[i] if i in labels_rtdetr else "-"
        
        writer.writerow([i, lbl_frcnn, lbl_retina, lbl_yolo, lbl_rtdetr])

print(f"Fichier sauvegardé avec succès ! ({max_length} lignes écrites)")