import os

input_root = "AutoShot_TEST/Adaptive_AutoShot"
output_root = "AutoShot_TEST/Adaptive_AutoShot_bis"


os.makedirs(output_root, exist_ok=True)

# for folder in os.listdir(input_root):
#     input_folder = os.path.join(input_root, folder)
#     output_folder = os.path.join(output_root, folder)

#     if not os.path.isdir(input_folder):
#         continue

#     os.makedirs(output_folder, exist_ok=True)

input_folder = input_root
output_folder = output_root

for filename in os.listdir(input_folder):
    if not filename.endswith(".txt"):
        continue

    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, filename)

    with open(input_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    new_lines = []
    for i in range(len(lines) - 1):
        _, b = lines[i].split()
        c, _ = lines[i + 1].split()
        new_lines.append(f"{b} {c}")

    with open(output_path, "w") as f:
        f.write("\n".join(new_lines))


