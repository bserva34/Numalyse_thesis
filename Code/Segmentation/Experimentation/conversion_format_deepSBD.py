import os

input_root = "BBC_Fade/DeepSBD_bis"
output_root = "BBC_Fade/DeepSBD_bis_bis"

# input_root = "BBC_FadeBlack/DeepSBD_bis"
# output_root = "BBC_FadeBlack/DeepSBD_bis_bis"

# input_root = "BBC_FadeBlack_mirror/DeepSBD_bis"
# output_root = "BBC_FadeBlack_mirror/DeepSBD_bis_bis"

os.makedirs(output_root, exist_ok=True)

# for folder in os.listdir(input_root):
#     input_folder = os.path.join(input_root, folder)
#     output_folder = os.path.join(output_root, folder)

#     if not os.path.isdir(input_folder):
#         continue

#     os.makedirs(output_folder, exist_ok=True)

input_folder=input_root
output_folder=output_root

for filename in os.listdir(input_folder):
    if not filename.endswith(".txt"):
        continue

    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, filename)

    kept_lines = []

    with open(input_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Exemple de ligne :
            # 7632    7847 : 1
            left, label = line.split(":")
            label = int(label.strip())

            if label > 0:
                parts = left.split()
                start = parts[0]
                end = parts[1]
                kept_lines.append(f"{start} {end}")

    with open(output_path, "w") as f:
        f.write("\n".join(kept_lines))
