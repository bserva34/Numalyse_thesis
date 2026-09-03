import csv
import re

# Définition des colonnes cibles dans le CSV
METHODS = ["Our_method", "Pei", "Middle_frame", "Zhang_et_al", "Chen_et_al"]
HEADER = ["Plan", "Expert"] + METHODS

# Nom des fichiers
input_filename = "questionnaire_1_expert.txt"
output_filename = "expert_vote.csv"

rows = []
current_plan = None

with open(input_filename, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        # Détection d'un nouveau Plan (ex: Plan1, Plan2)
        if line.startswith("Plan"):
            current_plan = line.split()[0] # Récupère juste "Plan1" au lieu de "Plan1 "
            continue

        # Détection et traitement des lignes Experts
        if line.startswith("Expert"):
            # Séparer l'expert du reste des méthodes (gère la tabulation ou plusieurs espaces)
            parts = re.split(r'\t+', line)
            
            # Si pas de tabulation, essayer de séparer après "Expert X"
            if len(parts) == 1:
                match = re.match(r'(Expert \d+)\s*(.*)', line)
                if match:
                    expert_name, methods_raw = match.groups()
                else:
                    continue
            else:
                expert_name = parts[0].strip()
                methods_raw = parts[1].strip() if len(parts) > 1 else ""

            # Transformer "Expert 1" en "E1"
            expert_num = expert_name.split()[1]
            expert_id = f"E{expert_num}"

            # Extraire et nettoyer la liste des méthodes citées
            # Enlève les espaces inutiles et filtre les éléments vides (ex: virgule finale)
            present_methods = [m.strip() for m in methods_raw.split(",") if m.strip()]

            # Créer la ligne de données avec 1 si présent, sinon 0
            row = [current_plan, expert_id]
            for method in METHODS:
                if method in present_methods:
                    row.append(1)
                else:
                    row.append(0)
            
            rows.append(row)

# Écriture du fichier CSV de sortie
with open(output_filename, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(HEADER)
    writer.writerows(rows)

print(f"Le fichier '{output_filename}' a été généré avec succès !")