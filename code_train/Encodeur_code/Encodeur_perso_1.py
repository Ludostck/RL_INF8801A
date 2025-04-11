import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision.models import resnet18  # <-- Import ResNet50
from tqdm import tqdm

# === PARAMÈTRES ===
IMG_DIR = "DataSet/dog_unique/img"
CSV_PATH = "code_train/Encodeur_code/bbox_iou_dataset.csv"
BEST_MODEL_PATH = "code_train/Encodeur_code/best_iou_encoder.pt"
BATCH_SIZE = 32
NUM_EPOCHS = 50
LR = 1e-4
INPUT_SIZE = (224, 224)
PADDING = 16
SEED = 42

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# === MODÈLE ===
class IoUEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Charger ResNet50 pré-entraîné
        base_model = resnet18(pretrained=True)
        # Si tu préfères ResNet101 : 
        # from torchvision.models import resnet101
        # base_model = resnet101(pretrained=True)

        # Tronquer la dernière couche (fc) pour ne garder que les features
        # Pour ResNet50/101, la sortie du bloc final est de taille 2048
        self.encoder = nn.Sequential(*list(base_model.children())[:-1])  # output: (B, 2048, 1, 1)

        # Tête de régression (dimension d'entrée = 2048)
        self.head = nn.Sequential(
            nn.Flatten(),               # (B, 2048)
            nn.Linear(512, 128),       # Ajuste selon la taille souhaitée
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()                # On borne la sortie entre 0 et 1
        )

    def forward(self, x):
        x = self.encoder(x)  # (B, 2048, 1, 1)
        x = self.head(x)     # (B, 1)
        return x.squeeze(1)  # (B,) pour correspondre à la shape (batch_size)

# === DATASET ===
class BBoxIoUDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.df = dataframe
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['image'])
        bbox = eval(row['bbox'])
        iou = row['iou']

        img = Image.open(img_path).convert('RGB')
        W, H = img.size

        # Crop avec padding
        x1 = max(0, int(bbox[0]) - PADDING)
        y1 = max(0, int(bbox[1]) - PADDING)
        x2 = min(W, int(bbox[2]) + PADDING)
        y2 = min(H, int(bbox[3]) + PADDING)


        crop = img.crop((x1, y1, x2, y2))

        if self.transform:
            crop = self.transform(crop)

        return crop, torch.tensor(iou, dtype=torch.float32)

# === TRANSFORMATIONS ===
transform = transforms.Compose([
    transforms.Resize(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === DONNÉES ===
df = pd.read_csv(CSV_PATH)
train_df, val_df = train_test_split(df, test_size=0.1, random_state=SEED)

train_dataset = BBoxIoUDataset(train_df, IMG_DIR, transform=transform)
val_dataset = BBoxIoUDataset(val_df, IMG_DIR, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# === ENTRAÎNEMENT ===
model = IoUEncoder().to(device)
criterion = nn.SmoothL1Loss()
optimizer = optim.Adam(model.parameters(), lr=LR)

def evaluate(model, loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = model(x)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return mae, rmse

best_val_mae = float("inf")

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=False)
    for x, y in loop:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        loop.set_postfix(loss=loss.item())

    train_loss = running_loss / len(train_loader.dataset)
    val_mae, val_rmse = evaluate(model, val_loader)

    print(f"✅ Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} "
          f"| Val MAE: {val_mae:.4f} | Val RMSE: {val_rmse:.4f}")

    if val_mae < best_val_mae:
        best_val_mae = val_mae
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"💾 Nouveau meilleur modèle sauvegardé (MAE: {val_mae:.4f})")
