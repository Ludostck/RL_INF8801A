import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18

# =============================================================================
# 1) Chemin d'accès au modèle pré-entraîné
# =============================================================================
MODEL_PATH = "code_train/Encodeur_code/best_iou_encoder.pt"

# =============================================================================
# 2) Modèle complet (encodeur + tête de régression) pour l'estimation de l'IoU
# =============================================================================
class IoUEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Le modèle ResNet18 pré-entraîné sur ImageNet est obtenu
        base_model = resnet18(pretrained=True)
        # Les couches jusqu'à l'avant-dernière sont regroupées pour extraire les features
        self.encoder = nn.Sequential(*list(base_model.children())[:-1])  # Sortie: (B, 512, 1, 1)
        # La tête de régression se charge d'estimer l'IoU à partir des features
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Les features passent par l'encodeur,
        # puis la tête de régression produit une estimation, avec suppression de la dimension superflue.
        x = self.encoder(x)
        return self.head(x).squeeze(1)

# =============================================================================
# 3) Chargement du modèle complet et des poids entraînés
# =============================================================================
full_model = IoUEncoder()
full_model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))

# =============================================================================
# 4) Extraction de l'encodeur seul (sans la tête), désigné par cnn, en mode évaluation
# =============================================================================
cnn = nn.Sequential(
    full_model.encoder,  # La sortie attendue est de forme (B, 512, 1, 1)
    nn.Flatten()         # Transformation en (B, 512)
)
cnn.eval()

# =============================================================================
# 5) Transformation appliquée aux images
# =============================================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# =============================================================================
# 6) Dueling Q‑Network
# =============================================================================
class QNetwork(nn.Module):
    """
    Architecture Dueling du Q‑Network.

    Args:
        input_dim (int): Dimension des caractéristiques d'entrée (ex. sortie du CNN).
        history_dim (int): Dimension des caractéristiques de l'historique (ex. vecteurs one‑hot).
        num_actions (int): Nombre total d'actions possibles.
    """
    def __init__(self, input_dim, history_dim, num_actions):
        super(QNetwork, self).__init__()
        
        # La dimension totale résulte de la concaténation des features d'entrée et d'historique
        total_input_dim = input_dim + history_dim
        
        # La couche commune des features opère la fusion initiale
        self.feature_layer = nn.Sequential(
            nn.Linear(total_input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 768),
            nn.ReLU(),
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU()
        )
        
        # Stream Value pour l'estimation de V(s)
        self.value_head = nn.Linear(512, 1)
        
        # Stream Advantage pour l'estimation de A(s, a)
        self.advantage_head = nn.Linear(512, num_actions)
        
        # Les poids des couches se voient attribuer une distribution Xavier Uniform ;
        # les biais sont mis à zéro.
        self.initialize_weights()
    
    def initialize_weights(self):
        """
        Les poids des couches linéaires se voient attribuer une distribution Xavier Uniform,
        et les biais sont mis à zéro.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, input_features, history_features):
        """
        Le passage avant du Dueling Q‑Network s'effectue de la manière suivante :

        Args:
            input_features  (torch.Tensor): Tenseur de taille (batch_size, input_dim)
            history_features (torch.Tensor): Tenseur de taille (batch_size, history_dim)

        Renvoie:
            torch.Tensor: Q‑valeurs de taille (batch_size, num_actions).
        """
        # La concaténation des features d'entrée et de l'historique s'effectue ici
        x = torch.cat((input_features, history_features), dim=1)  # (batch_size, total_input_dim)
        
        # La couche commune extrait des caractéristiques de dimension 512
        features = self.feature_layer(x)  # (batch_size, 512)
        
        # La tête Value produit V(s)
        value = self.value_head(features)  # (batch_size, 1)
        
        # La tête Advantage produit A(s, a)
        advantage = self.advantage_head(features)  # (batch_size, num_actions)
        
        # La combinaison procède en soustrayant la moyenne d'avantage afin de distinguer V(s) et A(s, a)
        advantage_mean = advantage.mean(dim=1, keepdim=True)
        q_values = value + (advantage - advantage_mean)
        
        return q_values
