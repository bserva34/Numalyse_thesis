import torch
from transformers import CLIPModel

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained(
    "openai/clip-vit-large-patch14"
).to(device)

print(next(model.parameters()).device)