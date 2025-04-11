import os
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import transforms
from PIL import Image
import pandas as pd
import torch.nn as nn
import xml.etree.ElementTree as ET
from torchvision.models import resnet50
# === PARAMÈTRES ===
IMG_DIR = "DataSet/dog_unique_test/img"
ANN_DIR = "DataSet/dog_unique_test/annotations"
CSV_PATH = "code_train/Encodeur_code/bbox_iou_dataset_test.csv"
MODEL_PATH = "code_train/Encodeur_code/best_iou_encoder_2.pt"
INPUT_SIZE = (224, 224)
PADDING = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === MODÈLE ===
# === MODÈLE ===
class IoUEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Charger ResNet50 pré-entraîné
        base_model = resnet50(pretrained=True)
        # Si tu préfères ResNet101 : 
        # from torchvision.models import resnet101
        # base_model = resnet101(pretrained=True)

        # Tronquer la dernière couche (fc) pour ne garder que les features
        # Pour ResNet50/101, la sortie du bloc final est de taille 2048
        self.encoder = nn.Sequential(*list(base_model.children())[:-1])  # output: (B, 2048, 1, 1)

        # Tête de régression (dimension d'entrée = 2048)
        self.head = nn.Sequential(
            nn.Flatten(),               # (B, 2048)
            nn.Linear(2048, 512),       # Ajuste selon la taille souhaitée
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()                # On borne la sortie entre 0 et 1
        )

    def forward(self, x):
        x = self.encoder(x)  # (B, 2048, 1, 1)
        x = self.head(x)     # (B, 1)
        return x.squeeze(1)  # (B,) pour correspondre à la shape (batch_size)

# === CHARGEMENT DU MODÈLE ===
model = IoUEncoder().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# === TRANSFORMATIONS ===
transform = transforms.Compose([
    transforms.Resize(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === DONNÉES ===
df = pd.read_csv(CSV_PATH)
sample_df = df.sample(6).reset_index(drop=True)

# === AFFICHAGE ===
fig, axs = plt.subplots(2, 3, figsize=(18, 10))

for i, ax in enumerate(axs.flat):
    row = sample_df.iloc[i]
    img_path = os.path.join(IMG_DIR, row['image'])
    ann_path = os.path.join(ANN_DIR, os.path.splitext(row['image'])[0] + ".xml")
    fake_bbox = eval(row['bbox'])
    true_iou = row['iou']

    img = Image.open(img_path).convert("RGB")
    W, H = img.size

    # === Prédiction IoU
    x1 = max(0, int(fake_bbox[0]) - PADDING)
    y1 = max(0, int(fake_bbox[1]) - PADDING)
    x2 = min(W, int(fake_bbox[2]) + PADDING)
    y2 = min(H, int(fake_bbox[3]) + PADDING)
    crop = img.crop((x1, y1, x2, y2))
    input_tensor = transform(crop).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred_iou = model(input_tensor).item()

    # === Affichage
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(f"True IoU: {true_iou:.2f} | Pred: {pred_iou:.2f}", fontsize=10)

    # Fake bbox (rouge)
    fx1, fy1, fx2, fy2 = fake_bbox
    ax.add_patch(patches.Rectangle(
        (fx1, fy1), fx2 - fx1, fy2 - fy1,
        linewidth=2, edgecolor='r', facecolor='none', label='Fake BBOX'
    ))

    # True bbox (verte) depuis XML
    if os.path.exists(ann_path):
        tree = ET.parse(ann_path)
        root = tree.getroot()
        # Liste des classes de chiens
    dog_classes = [
        "Affenpinscher", "Afghan_hound", "African_hunting_dog", "Airedale", "American_Staffordshire_terrier",
        "Appenzeller", "Australian_terrier", "Basenji", "Basset", "Beagle", "Bedlington_terrier",
        "Bernese_mountain_dog", "Black-and-tan_coonhound", "Blenheim_spaniel", "Bloodhound", "Bluetick",
        "Border_collie", "Border_terrier", "Borzoi", "Boston_bull", "Bouvier_des_Flandres", "Boxer",
        "Brabancon_griffon", "Briard", "Brittany_spaniel", "Bull_mastiff", "Cairn", "Cardigan", "Chesapeake_Bay_retriever",
        "Chihuahua", "Chow", "Clumber", "Cocker_spaniel", "Collie", "Curly-coated_retriever", "Dandie_Dinmont",
        "Dhole", "Dingo", "Doberman", "English_foxhound", "English_setter", "English_springer", "EntleBucher",
        "Eskimo_dog", "Flat-coated_retriever", "French_bulldog", "German_shepherd", "German_short-haired_pointer",
        "Giant_schnauzer", "Golden_retriever", "Gordon_setter", "Great_Dane", "Great_Pyrenees", "Greater_Swiss_Mountain_dog",
        "Groenendael", "Ibizan_hound", "Irish_setter", "Irish_terrier", "Irish_water_spaniel", "Irish_wolfhound",
        "Italian_greyhound", "Japanese_spaniel", "Keeshond", "Kelpie", "Kerry_blue_terrier", "Komondor",
        "Kuvasz", "Labrador_retriever", "Lakeland_terrier", "Leonberg", "Lhasa", "Malamute", "Malinois",
        "Maltese_dog", "Mexican_hairless", "Miniature_pinscher", "Miniature_poodle", "Miniature_schnauzer",
        "Newfoundland", "Norfolk_terrier", "Norwegian_elkhound", "Norwich_terrier", "Old_English_sheepdog",
        "Otterhound", "Papillon", "Pekinese", "Pembroke", "Petit_basset_griffon_Vendeen", "Pharaoh_hound",
        "Plott", "Pointer", "Pomeranian", "Pug", "Redbone", "Rhodesian_ridgeback", "Rottweiler",
        "Saint_Bernard", "Saluki", "Samoyed", "Schipperke", "Scotch_terrier", "Scottish_deerhound",
        "Sealyham_terrier", "Shetland_sheepdog", "Shih-Tzu", "Siberian_husky", "Silky_terrier", "Soft-coated_wheaten_terrier",
        "Staffordshire_bullterrier", "Standard_poodle", "Standard_schnauzer", "Sussex_spaniel", "Tibetan_mastiff",
        "Tibetan_terrier", "Toy_poodle", "Toy_terrier", "Vizsla", "Walker_hound", "Weimaraner", "Welsh_springer_spaniel",
        "West_Highland_white_terrier", "Whippet", "Wire-haired_fox_terrier", "Yorkshire_terrier", "dog"
    ]

    # Recherche de la vraie BBOX de type "chien"
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        if name in dog_classes:
            bndbox = obj.find("bndbox")
            tx1 = float(bndbox.find("xmin").text)
            ty1 = float(bndbox.find("ymin").text)
            tx2 = float(bndbox.find("xmax").text)
            ty2 = float(bndbox.find("ymax").text)

            ax.add_patch(patches.Rectangle(
                (tx1, ty1), tx2 - tx1, ty2 - ty1,
                linewidth=2, edgecolor='g', facecolor='none', label='True BBOX'
            ))
            break



          

plt.tight_layout()
plt.show()
