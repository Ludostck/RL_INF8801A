import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18

# =============================================================================
# 1) PARAMÈTRES : Chemin vers le modèle pré-entraîné
# =============================================================================
MODEL_PATH = "code_train/Encodeur_code/best_iou_encoder.pt"

# =============================================================================
# 2) Définition du modèle complet (encodeur + tête) pour l'estimation de l'IoU
# =============================================================================
class IoUEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Le modèle ResNet18 pré-entraîné sur ImageNet est chargé
        base_model = resnet18(pretrained=True)
        # Les couches jusqu'à l'avant-dernière sont regroupées pour extraire les features
        self.encoder = nn.Sequential(*list(base_model.children())[:-1])  # Sortie : (B, 512, 1, 1)
        # La tête de régression permet d'estimer l'IoU à partir des features
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Passage par l'encodeur pour extraire les caractéristiques
        x = self.encoder(x)
        # La tête de régression produit l'estimation de l'IoU avec suppression de la dimension superflue
        return self.head(x).squeeze(1)

# =============================================================================
# 3) Chargement du modèle complet et des poids entraînés
# =============================================================================
full_model = IoUEncoder()
full_model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))

# =============================================================================
# 4) Extraction de l'encodeur seul (sans la tête), nommé cnn, et mise en mode évaluation
# =============================================================================
cnn = nn.Sequential(
    full_model.encoder,  # La sortie attendue est de forme (B, 512, 1, 1)
    nn.Flatten()         # Transformation en (B, 512)
)
cnn.eval()

# =============================================================================
# 5) Définition de la transformation appliquée aux images
# =============================================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Redimensionnement à 224x224
    transforms.ToTensor(),          # Conversion en tenseur
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalisation avec les moyennes d'ImageNet
                         std=[0.229, 0.224, 0.225])   # et les écarts-types d'ImageNet
])

# =============================================================================
# 6) Q‑Network standard
# =============================================================================
class QNetwork(nn.Module):
    """
    Q‑Network standard.

    Args:
        input_dim (int): Dimension des caractéristiques d'entrée.
        history_dim (int): Dimension des caractéristiques d'historique.
        num_actions (int): Nombre d'actions possibles.
    """
    def __init__(self, input_dim, history_dim, num_actions):
        super(QNetwork, self).__init__()
        
        # La dimension totale correspond à la somme des dimensions d'entrée et d'historique
        total_input_dim = input_dim + history_dim  # par ex. input_dim + 90
        
        # La couche de features regroupe deux couches linéaires de 1024 neurones chacune
        self.feature_layer = nn.Sequential(
            nn.Linear(total_input_dim, 1024),  # Conversion : total_input_dim → 1024
            nn.ReLU(),
            nn.Linear(1024, 1024),             # 1024 → 1024
            nn.ReLU()
        )
        
        # La couche de sortie produit les Q‑values pour chaque action
        self.output_layer = nn.Linear(1024, num_actions)  # 1024 → num_actions
        
        # Les poids se voient initialisés
        self.initialize_weights()
        
    def initialize_weights(self):
        """
        Les couches linéaires se voient initialisées avec une répartition Xavier Uniform
        et leurs biais se voient mis à zéro.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, input_features, history_features):
        """
        Le passage avant du Q‑Network procède ainsi :

        Args:
            input_features (torch.Tensor): Tenseur de forme (batch_size, input_dim).
            history_features (torch.Tensor): Tenseur de forme (batch_size, history_dim).

        Renvoie:
            torch.Tensor: Q‑values pour chaque action, de forme (batch_size, num_actions).
        """
        # La concaténation des caractéristiques d'entrée et de l'historique se réalise ici
        x = torch.cat((input_features, history_features), dim=1)  # (batch_size, total_input_dim)
        
        # Passage dans la couche de features
        features = self.feature_layer(x)  # (batch_size, 1024)
        
        # Passage dans la couche de sortie pour obtenir les Q‑values
        q_values = self.output_layer(features)  # (batch_size, num_actions)
            
        return q_values
