import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50

# =============================================================================
# 1) Chemin d'accès au modèle pré-entraîné pour l'encodeur IoU
# =============================================================================
MODEL_PATH = "code_train/Encodeur_code/best_iou_encoder_2.pt"

# Activation du benchmark cuDNN pour optimiser les performances
torch.backends.cudnn.benchmark = True

# =============================================================================
# 2) Modèle IoUEncoder basé sur ResNet50 avec tête de régression
# =============================================================================
class IoUEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Le modèle ResNet50 pré-entraîné sur ImageNet est récupéré
        base_model = resnet50(pretrained=True)
        # (Option : pour utiliser ResNet101, décommenter les lignes correspondantes)
        
        # Les dernières couches (fc) sont exclues afin de ne conserver que les features,
        # la sortie du bloc final présente une dimension de 2048.
        self.encoder = nn.Sequential(*list(base_model.children())[:-1])  # Sortie : (B, 2048, 1, 1)

        # La tête de régression ajuste la sortie et borne l'estimation entre 0 et 1
        self.head = nn.Sequential(
            nn.Flatten(),               # Aplatissement vers (B, 2048)
            nn.Linear(2048, 512),       # Ajustement de la dimension
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()                # La sortie est bornée entre 0 et 1
        )

    def forward(self, x):
        # Les features traversent l'encodeur (B, 2048, 1, 1)
        x = self.encoder(x)
        # La tête de régression fournit la prédiction d'IoU (B, 1) avec suppression de la dimension superflue
        x = self.head(x)
        return x.squeeze(1)  # Forme finale (B,)
    
# =============================================================================
# 3) Chargement du modèle complet et des poids entraînés
# =============================================================================
full_model = IoUEncoder()
full_model.load_state_dict(torch.load(MODEL_PATH, map_location='cuda'))

# =============================================================================
# 4) Extraction de l'encodeur seul (sans la tête), désigné par cnn, et passage en mode évaluation
# =============================================================================
cnn = nn.Sequential(
    full_model.encoder,  # La sortie obtenue sera de forme (B, 2048, 1, 1)
    nn.Flatten()         # Transformation en (B, 2048)
)
cnn.eval()

# Définition de la taille d'entrée des images
INPUT_SIZE = (224, 224)

# =============================================================================
# 5) Transformation appliquée aux images
# =============================================================================
transform = transforms.Compose([
    transforms.Resize(INPUT_SIZE),
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
        
        # La couche commune des features opère la fusion
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
        
        # Stream Value pour produire V(s)
        self.value_head = nn.Linear(512, 1)
        
        # Stream Advantage pour produire A(s, a)
        self.advantage_head = nn.Linear(512, num_actions)
        
        # Les poids se voient attribuer une répartition Xavier Uniform et les biais mis à zéro
        self.initialize_weights()
    
    def initialize_weights(self):
        """
        Les poids des couches linéaires se voient attribuer une répartition Xavier Uniform,
        et les biais se voient mis à zéro.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, input_features, history_features):
        """
        Le passage avant du Dueling Q‑Network s'effectue comme suit :

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
        
        # La combinaison procède en soustrayant la moyenne d'avantage afin de distinguer V(s) et A(s,a)
        advantage_mean = advantage.mean(dim=1, keepdim=True)
        q_values = value + (advantage - advantage_mean)
        
        return q_values
