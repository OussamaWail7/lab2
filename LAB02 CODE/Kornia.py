import cv2
import torch
import numpy as np
from kornia.contrib import ImageStitcher
import matplotlib.pyplot as plt

# Chemins d'accès des images que vous souhaitez charger
image_paths = ["ENV4/IMG3.jpg", "ENV4/IMG4.jpg"]

# Charger les images en tant que tenseurs PyTorch
imgs = []
for path in image_paths:
    img = cv2.imread(path)
    if img is not None:
        # Convertir l'image en format RGB et normaliser les valeurs des pixels
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.tensor(img_rgb, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to('cuda') / 255.0
        imgs.append(img_tensor)
    else:
        print("Impossible de charger l'image à partir de", path)

# Initialiser l'objet ImageStitcher avec le matcher LoFTR et l'estimateur RANSAC
matcher = LoFTR(pretrained='outdoor').cuda()
IS = ImageStitcher(matcher, estimator='ransac').cuda()

# Exécuter le stitching des deux images
with torch.no_grad():
    out = IS(*imgs)

# Convertir l'image panoramique en format numpy
panorama = out.squeeze(0).permute(1, 2, 0).cpu().numpy()

# Afficher l'image panoramique
plt.imshow(panorama)
plt.axis('off')
plt.show()
