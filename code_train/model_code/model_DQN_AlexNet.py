import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import models
from torchvision.models import AlexNet_Weights  # Poids pré-entraînés importés

# =============================================================================
# 1) CNN pré-entraîné (AlexNet) destiné à l'extraction de caractéristiques
# =============================================================================
def load_pretrained_cnn():
    """
    Le modèle AlexNet pré-entraîné sur ImageNet est utilisé jusqu'à la couche fc6,
    ce qui permet d'obtenir des caractéristiques exploitables.

    Renvoie:
        torch.nn.Sequential: Modèle extracteur de caractéristiques.
    """
    # Le modèle AlexNet pré-entraîné sur ImageNet est récupéré
    model = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)

    # On constitue un extracteur en regroupant :
    # • les couches convolutionnelles (model.features)
    # • l'opération d'aplatissement (Flatten)
    # • la couche fc6 (model.classifier[1])
    feature_extractor = torch.nn.Sequential(
        *list(model.features),
        torch.nn.Flatten(),
        model.classifier[1],
    )

    # Le modèle extracteur passe en mode évaluation
    feature_extractor.eval()

    return feature_extractor

# Le modèle CNN est chargé via la fonction ci-dessus
cnn = load_pretrained_cnn()

# =============================================================================
# 2) Transformation appliquée aux images pour la normalisation
# =============================================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Les images sont redimensionnées en 224x224
    transforms.ToTensor(),          # La conversion d'une image PIL en tenseur est effectuée
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # La normalisation se base sur les moyennes d'ImageNet
                         std=[0.229, 0.224, 0.225]),   # et sur les écarts-types d'ImageNet
])

# =============================================================================
# 3) Dueling Q‑Network
# =============================================================================
class QNetwork(nn.Module):
    """
    Architecture Dueling du Q‑Network.

    Args:
        input_dim (int): Dimension des caractéristiques d'entrée (ex. la sortie du CNN).
        history_dim (int): Dimension des caractéristiques de l'historique (ex. vecteurs one‑hot).
        num_actions (int): Nombre total d'actions possibles.
    """
    def __init__(self, input_dim, history_dim, num_actions):
        super(QNetwork, self).__init__()
        
        # La dimension totale correspond à la concaténation des caractéristiques d'entrée et d'historique
        total_input_dim = input_dim + history_dim
        
        # La couche de features commune opère la fusion
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
        
        # Stream Value (V(s)) pour estimer la valeur de l'état
        self.value_head = nn.Linear(512, 1)
        
        # Stream Advantage (A(s, a)) pour estimer l'avantage de chaque action
        self.advantage_head = nn.Linear(512, num_actions)
        
        # Les poids des couches linéaires se voient attribuer une répartition Xavier Uniform ;
        # les biais se voient mis à zéro.
        self.initialize_weights()
    
    def initialize_weights(self):
        """
        Les poids des couches linéaires se voient attribuer une répartition Xavier Uniform
        et les biais se voient mis à zéro.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, input_features, history_features):
        """
        Le passage avant du Dueling Q‑Network procède comme suit :

        Args:
            input_features  (torch.Tensor): Tenseur de taille (batch_size, input_dim)
            history_features (torch.Tensor): Tenseur de taille (batch_size, history_dim)

        Renvoie:
            torch.Tensor: Q‑valeurs de taille (batch_size, num_actions).
        """
        # La concaténation des caractéristiques d'entrée et de l'historique se réalise ici
        x = torch.cat((input_features, history_features), dim=1)  # (batch_size, total_input_dim)
        
        # La couche commune extrait des features de dimension 512
        features = self.feature_layer(x)  # (batch_size, 512)
        
        # La tête Value calcule V(s)
        value = self.value_head(features)  # (batch_size, 1)
        
        # La tête Advantage calcule A(s, a)
        advantage = self.advantage_head(features)  # (batch_size, num_actions)
        
        # La combinaison se fait en soustrayant la moyenne d'avantage pour distinguer V(s) et A(s,a)
        advantage_mean = advantage.mean(dim=1, keepdim=True)
        q_values = value + (advantage - advantage_mean)
        
        return q_values
