import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import re

def numerical_sort(value): # Fonction de tri numérique des noms de fichiers
    numbers = re.findall(r'\d+', value)
    return int(numbers[0]) if numbers else 0

transform = transforms.Compose([
   transforms.Resize((224, 224)),  # Taille d'entrée fixe pour EfficientNet-B0
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
# Définir le Dataset pour le dossier de test
class TestDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_names = sorted(os.listdir(img_dir), key=numerical_sort)  # Tri numérique des noms de fichiers

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_name
    
test_dir = "test"  # Dossier de test, faut modifier si nécessaire
test_dataset = TestDataset(test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

if __name__ == '__main__':
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu" # S'assurer d'utiliser le bon device
    print(device)
    classes = ['bread', 'dairy', 'dessert', 'egg', 'fried', 'meat', 'noodles-pasta', 'rice', 'seafood', 'soup', 'vegetable-fruit']
    pred_dict = {}
    # Load model and set to eval mode, on utilise EfficientNet-B0 car c'est plus performant
    model = efficientnet_b0()
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 11)
    model.load_state_dict(torch.load("modelCNN.pth", weights_only=True))
    model.to(device)
    model.eval()

    with torch.no_grad(): # Commencer les prédictions
        for images, img_names in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            for i, img_name in enumerate(img_names):
                idx = test_dataset.img_names.index(img_name)  # giữ thứ tự index
                pred_dict[str(idx)] = classes[preds[i]]
    # -------------------------
    # Sauvegarder en JSON
    # -------------------------
    with open("predictions.json", "w") as f:
        json.dump(pred_dict, f, indent=4)

    print("Done! JSON file created.")