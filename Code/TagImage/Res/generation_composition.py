import os
import random
import csv
from PIL import Image, ImageDraw, ImageFont

def get_prefix(filename):
    lower_name = filename.lower()
    if "_keyframe" in lower_name:
        index = lower_name.find("_keyframe")
        return filename[:index]
    return os.path.splitext(filename)[0]

def get_robust_font_thin(font_size):
    # Liste de polices configurées spécifiquement sur leurs déclinaisons "Light" ou "Thin"
    font_paths = [
        # Windows
        "segoeuil.ttf",                       # Segoe UI Light (Très fin et moderne)
        "corbel.ttf",                         # Corbel (Plus fin d'origine qu'Arial)
        
        # macOS
        "/System/Library/Fonts/HelveticaLight.ttf", 
        "/System/Library/Fonts/Supplemental/Futura Medium.otf", # Tracé géométrique assez fin
        
        # Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-ExtraLight.ttf",
        
        # Fallbacks classiques
        "arial.ttf",
        "Arial.ttf"
    ]
    
    for path in font_paths:
        try:
            return ImageFont.truetype(path, font_size)
        except IOError:
            continue
            
    print("Information : Impossible de trouver une police fine. Utilisation de la police par défaut.")
    return ImageFont.load_default()

def combine_images_by_prefix(parent_folder, output_folder, log_file_path, max_cell_width=800, max_cell_height=600):
    # 1. Identifier les sous-dossiers
    subfolders = [f for f in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, f))]
    
    if len(subfolders) != 6:
        print(f"Attention : Le script attend 6 sous-dossiers. Trouvé : {len(subfolders)}.")
        if len(subfolders) == 0:
            print("Aucun sous-dossier trouvé. Arrêt du script.")
            return

    print(f"Sous-dossiers détectés : {subfolders}")

    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')

    # 2. Cartographier les fichiers de chaque dossier par leur préfixe
    prefix_map = {}
    for sub in subfolders:
        sub_path = os.path.join(parent_folder, sub)
        files = [f for f in os.listdir(sub_path) if f.lower().endswith(valid_extensions)]
        for f in files:
            prefix = get_prefix(f)
            if prefix not in prefix_map:
                prefix_map[prefix] = {}
            prefix_map[prefix][sub] = f

    # Filtrer pour garder uniquement les préfixes présents dans les 6 sous-dossiers
    common_prefixes = [pref for pref, mapping in prefix_map.items() if len(mapping) == len(subfolders)]

    if not common_prefixes:
        print("Aucune correspondance de préfixe (avant '_keyframe') n'a été trouvée dans les 6 sous-dossiers.")
        return

    print(f"Nombre de groupes d'images trouvés à combiner : {len(common_prefixes)}")

    # Créer le dossier de sortie s'il n'existe pas
    os.makedirs(output_folder, exist_ok=True)

    header = ["Prefixe_Commun"] + [f"Image {i+1}" for i in range(len(subfolders))]
    log_data = []

    # 3. Traiter chaque groupe d'images
    for prefix in common_prefixes:
        sources = []
        for sub in subfolders:
            filename = prefix_map[prefix][sub]
            sources.append((sub, os.path.join(parent_folder, sub, filename)))
        
        # Mélange aléatoire
        random.shuffle(sources)

        # Charger et redimensionner les images pour uniformiser leur taille
        loaded_images = []
        for sub, path in sources:
            try:
                img = Image.open(path)
                img = img.convert("RGB") # Nettoyage de la transparence d'origine
                img.thumbnail((max_cell_width, max_cell_height), Image.Resampling.LANCZOS)
                loaded_images.append((sub, img))
            except Exception as e:
                print(f"Erreur lors de l'ouverture/redimensionnement de {path}: {e}")

        if len(loaded_images) < len(subfolders):
            print(f"Saut du groupe '{prefix}' suite à une erreur de traitement.")
            continue

        # Taille de grille uniforme
        cell_w = max(img.width for _, img in loaded_images)
        cell_h = max(img.height for _, img in loaded_images)

        cols = 3
        rows = 2
        
        grid_w = cell_w * cols
        grid_h = cell_h * rows

        # Créer l'image finale combinée
        combined_img = Image.new('RGB', (grid_w, grid_h), color=(0, 0, 0))
        draw = ImageDraw.Draw(combined_img)

        # Taille du chiffre ajustée à 45% (légèrement plus petit pour gagner en élégance)
        font_size = int(cell_h * 0.45)
        font = get_robust_font_thin(font_size)

        mapping = {"Prefixe_Commun": prefix}

        for index, (sub_name, img) in enumerate(loaded_images):
            c = index % cols
            r = index // cols

            x_offset = c * cell_w
            y_offset = r * cell_h

            # Centrer l'image d'origine dans sa case
            img_x = x_offset + (cell_w - img.width) // 2
            img_y = y_offset + (cell_h - img.height) // 2
            
            # Coller l'image sur la grille
            combined_img.paste(img, (img_x, img_y))

            # --- DESSINER LE CHIFFRE ROUGE FIN ---
            chiffre = str(index + 1)

            # Calcul du centrage précis
            if hasattr(draw, "textbbox"):
                bbox = draw.textbbox((0, 0), chiffre, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
            else:
                text_w, text_h = draw.textsize(chiffre, font=font) if hasattr(draw, "textsize") else (font_size, font_size)

            tx = img_x + (img.width - text_w) // 2
            ty = img_y + (img.height - text_h) // 2 - (text_h // 8)

            # Contour d'un seul pixel pour la lisibilité, sans épaissir le chiffre
            contour_color = (0, 0, 0)
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx != 0 or dy != 0:
                        draw.text((tx + dx, ty + dy), chiffre, fill=contour_color, font=font)

            # Dessiner le chiffre en rouge vif pur par-dessus
            draw.text((tx, ty), chiffre, fill=(255, 0, 0), font=font)

            mapping[f"Image {index + 1}"] = sub_name

        # Sauvegarde
        output_path = os.path.join(output_folder, f"{prefix}.png")
        combined_img.save(output_path)
        log_data.append(mapping)
        print(f"Image combinée créée : {output_path}")

    # 4. Écrire le fichier de correspondance (CSV)
    with open(log_file_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(log_data)
    
    print(f"\nFichier de correspondance généré avec succès : {log_file_path}")

if __name__ == "__main__":
    # --- CONFIGURATION DES CHEMINS ---
    DOSSIER_PARENT = "Q2_p1"
    DOSSIER_SORTIE = "Q2_combined"
    FICHIER_LOG = "Q2_combined/correspondances.csv"
    
    MAX_LARGEUR_IMAGE = 800  
    MAX_HAUTEUR_IMAGE = 600

    if os.path.exists(DOSSIER_PARENT):
        combine_images_by_prefix(
            DOSSIER_PARENT, 
            DOSSIER_SORTIE, 
            FICHIER_LOG, 
            max_cell_width=MAX_LARGEUR_IMAGE, 
            max_cell_height=MAX_HAUTEUR_IMAGE
        )
    else:
        print(f"Veuillez créer le dossier '{DOSSIER_PARENT}' et y placer vos 6 sous-dossiers.")