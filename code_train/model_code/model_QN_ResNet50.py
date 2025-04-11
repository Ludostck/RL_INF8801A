import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50

# =============================================================================
# 1) PARAMÈTRES : Chemin vers le modèle pré-entraîné pour l'encodeur IoU
# =============================================================================
MODEL_PATH = "code_train/Encodeur_code/best_iou_encoder_2.pt"

# Activation du benchmark cuDNN pour optimiser les performances
torch.backends.cudnn.benchmark = True

# =============================================================================
# 2) Définition du modèle IoUEncoder basé sur ResNet50 avec tête de régression
# =============================================================================
class IoUEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Le modèle ResNet50 pré-entraîné sur ImageNet est récupéré
        base_model = resnet50(pretrained=True)
        # (Option : pour utiliser ResNet101, décommenter les lignes correspondantes)
        
        # On tronque la dernière couche (fc) afin de conserver uniquement les features.
        # Pour ResNet50/101, la sortie du bloc final présente une dimension de 2048.
        self.encoder = nn.Sequential(*list(base_model.children())[:-1])  # Sortie : (B, 2048, 1, 1)

        # La tête de régression ajuste la dimension et borne la sortie entre 0 et 1
        self.head = nn.Sequential(
            nn.Flatten(),               # Aplatissement pour obtenir (B, 2048)
            nn.Linear(2048, 512),       # Transformation de la dimension
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()                # La sortie est bornée entre 0 et 1
        )

    def forward(self, x):
        # Passage par l'encodeur pour extraire les features
        x = self.encoder(x)  # (B, 2048, 1, 1)
        # Passage par la tête de régression, avec suppression de la dimension superflue
        x = self.head(x)     # (B, 1)
        return x.squeeze(1)  # Forme finale (B,)
    

# =============================================================================
# 3) Chargement du modèle complet et des poids entraînés
# =============================================================================
full_model = IoUEncoder()
full_model.load_state_dict(torch.load(MODEL_PATH, map_location='cuda'))

# =============================================================================
# 4) Extraction de l'encodeur seul (sans la tête), nommé cnn, et mise en mode évaluation
# =============================================================================
cnn = nn.Sequential(
    full_model.encoder,  # La sortie obtenue est de forme (B, 2049, 1, 1)
    nn.Flatten()         # Transformation en (B, 2049)
)
cnn.eval()

# =============================================================================
# 5) Définition de la taille d'entrée et des transformations appliquées aux images
# =============================================================================
INPUT_SIZE = (224, 224)

transform = transforms.Compose([
    transforms.Resize(INPUT_SIZE),  # Redimensionnement selon INPUT_SIZE
    transforms.ToTensor(),            # Conversion en tenseur
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
        
        # La dimension totale correspond à la somme de la dimension d'entrée et de l'historique
        total_input_dim = input_dim + history_dim  # par ex. input_dim + 90
        
        # La couche de features est composée de deux couches linéaires de 1024 neurones chacune
        self.feature_layer = nn.Sequential(
            nn.Linear(total_input_dim, 1024),  # Conversion : total_input_dim → 1024
            nn.ReLU(),
            nn.Linear(1024, 1024),             # 1024 → 1024
            nn.ReLU()
        )
        
        # La couche de sortie permet d'obtenir les Q‑values pour chaque action
        self.output_layer = nn.Linear(1024, num_actions)  # 1024 → num_actions
        
        # Les poids se voient initialisés
        self.initialize_weights()
        
    def initialize_weights(self):
        """
        Les couches linéaires se voient initialisées avec une répartition Xavier Uniform,
        et les biais se voient mis à zéro.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, input_features, history_features):
        """
        Le passage avant du Q‑Network se déroule comme suit :

        Args:
            input_features (torch.Tensor): Tenseur de forme (batch_size, input_dim).
            history_features (torch.Tensor): Tenseur de forme (batch_size, history_dim).

        Renvoie:
            torch.Tensor: Q‑values pour chaque action, de forme (batch_size, num_actions).
        """
        # Concaténation des caractéristiques d'entrée et de l'historique
        x = torch.cat((input_features, history_features), dim=1)  # (batch_size, total_input_dim)
        
        # Passage dans la couche de features
        features = self.feature_layer(x)  # (batch_size, 1024)
        
        # Passage dans la couche de sortie pour obtenir les Q‑values
        q_values = self.output_layer(features)  # (batch_size, num_actions)
            
        return q_values
