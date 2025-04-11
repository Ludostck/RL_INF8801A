import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import models
from torchvision.models import AlexNet_Weights  # Poids pré-entraînés importés

# =============================================================================
# 1) Chargement du CNN pré-entraîné (AlexNet) pour l'extraction de caractéristiques
# =============================================================================
def load_pretrained_cnn():
    """
    Le modèle AlexNet pré-entraîné sur ImageNet est récupéré jusqu'à la couche fc6,
    fournissant ainsi un extracteur de caractéristiques.

    Renvoie:
        torch.nn.Sequential: Modèle extracteur de caractéristiques.
    """
    # Le modèle AlexNet pré-entraîné sur ImageNet est chargé
    model = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)

    # On compose un modèle se terminant à la couche fc6 :
    # • Les couches de convolution (model.features)
    # • L'opération d'aplatissement (Flatten)
    # • La couche fc6 (model.classifier[1])
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
    transforms.Resize((224, 224)),  # Les images sont redimensionnées à 224x224
    transforms.ToTensor(),          # La conversion d'une image PIL en tenseur se fait ici
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # La normalisation se fait avec les moyennes d'ImageNet
                         std=[0.229, 0.224, 0.225]),   # et les écarts-types d'ImageNet
])

# =============================================================================
# 3) Q‑Network standard
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
        
        # La dimension totale résulte de la concaténation des features d'entrée et d'historique
        total_input_dim = input_dim + history_dim  # par ex. 4096 + 90 = 4186
        
        # La couche de feature comporte deux couches linéaires de 1024 neurones chacune
        self.feature_layer = nn.Sequential(
            nn.Linear(total_input_dim, 1024),  # 4186 → 1024
            nn.ReLU(),
            nn.Linear(1024, 1024),             # 1024 → 1024
            nn.ReLU()
        )
        
        # La couche de sortie permet d'obtenir les Q‑values pour chaque action
        self.output_layer = nn.Linear(1024, num_actions)  # 1024 → num_actions
        
        # Les poids des couches linéaires se voient initialisés
        self.initialize_weights()
        
    def initialize_weights(self):
        """
        Les couches linéaires sont initialisées avec une répartition Xavier Uniform,
        et les biais se voient mis à zéro.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, input_features, history_features):
        """
        Le passage avant du Q‑Network se déroule ainsi :

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
