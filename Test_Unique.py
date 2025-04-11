import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from collections import deque
from model_Base_CLIP import QNetwork, cnn, transform
from additional_functions import (
    ACTIONS, pil_to_tensor, calculate_iou, apply_action, calculate_reward,
    get_history_tensor, draw_cross, load_annotations, get_features
)
from tqdm import tqdm

# Set the device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Testing on device:", device)

# Define root directory and dataset paths
ROOT = ""
super_dataset_dir = os.path.join(ROOT, "TEST")
target_type = "dog"
images_dir_test = os.path.join(super_dataset_dir, target_type, "imgs")
annotations_dir_test = os.path.join(super_dataset_dir, target_type, "annotations")

# Create a list of test image IDs (without file extensions)
test_list = [os.path.splitext(f)[0] for f in os.listdir(images_dir_test) if f.endswith(".jpg")]

# Model parameters
input_dim = 768
history_dim = 90
num_actions = 9

# Path to the trained model
models_dir = os.path.join(ROOT, "models/Base/model_epoch_11_dog_BASE_CLIP.pth")  # Vérifiez ce chemin

IOU_THRESHOLD = 0.6

def test(q_network, cnn_model, test_list, images_dir_test, annotations_dir_test):
    # Listes pour suivre l'IoU final et le nombre d'actions par détection
    final_ious = []
    actions_per_detection = []
    
    # Compteurs pour les triggers
    total_triggers = 0         # Un trigger par groundtruth
    successful_triggers = 0    # Détections avec IoU >= IOU_THRESHOLD
    forced_trigger_count = 0   # Nombre de triggers forcés (aucun trigger sur les 4 coins)
    qnetwork_trigger_count = 0 # Nombre de triggers initiés par le QNetwork
    qnetwork_trigger_success = 0  # Parmi les triggers QNetwork, ceux avec IoU >= IOU_THRESHOLD

    # Pour les actions non-trigger
    total_non_trigger_actions = 0
    improved_actions = 0

    total_groundtruths = 0

    # Itération sur chaque image du jeu de test
    for image_id in tqdm(test_list, desc="Processing images"):
        annotation_file = os.path.join(annotations_dir_test, f"{image_id}.xml")
        image_file = os.path.join(images_dir_test, f"{image_id}.jpg")

        # Chargement et prétraitement de l'image
        image_pil = Image.open(image_file).convert("RGB")
        image_tensor = pil_to_tensor(image_pil).to(device)
        ground_truths = load_annotations(annotation_file)
        total_groundtruths += len(ground_truths)

        img_w, img_h = image_tensor.shape[2], image_tensor.shape[1]

        # Pré-calcul des 4 coins (bbox couvrant 75% de l'image)
        corners = [
            (0, 0, int(0.75 * img_w), int(0.75 * img_h)),                    # Haut-gauche
            (img_w - int(0.75 * img_w), 0, img_w, int(0.75 * img_h)),          # Haut-droit
            (0, img_h - int(0.75 * img_h), int(0.75 * img_w), img_h),          # Bas-gauche
            (img_w - int(0.75 * img_w), img_h - int(0.75 * img_h), img_w, img_h) # Bas-droit
        ]

        # Pour chaque groundtruth dans l'image
        for gt in ground_truths:
            detection_actions = 0  # Nombre d'actions pour cette détection
            history_deque = deque([torch.zeros(num_actions).to(device) for _ in range(10)], maxlen=10)

            # 1) Tentative sur l'image complète
            bbox_full = [0, 0, img_w, img_h]
            bbox = bbox_full.copy()
            triggered = False
            trigger_reward = 0.0  # Récompense au moment du trigger (optionnel)
            etat_final_1 = None

            for step in range(30):
                detection_actions += 1
                total_non_trigger_actions += 1

                current_iou = calculate_iou(bbox, gt)
                
                features = get_features(bbox, cnn_model, image_tensor, device, transform)
                history_tensor = get_history_tensor(history_deque)
                with torch.no_grad():
                    q_values = q_network(features, history_tensor)
                action_idx = torch.argmax(q_values).item()
                action = ACTIONS[action_idx]

                if action == "trigger":
                    triggered = True
                    trigger_reward = calculate_reward(current_iou, current_iou, action_idx)
                    qnetwork_trigger_count += 1
                    break
                else:
                    new_bbox = apply_action(bbox, action, img_w, img_h)
                    new_iou = calculate_iou(new_bbox, gt)
                    total_non_trigger_actions += 1
                    if new_iou > current_iou:
                        improved_actions += 1
                    bbox = new_bbox
                    one_hot_action = torch.zeros(num_actions).to(device)
                    one_hot_action[action_idx] = 1
                    history_deque.append(one_hot_action)
            etat_final_1 = bbox.copy()

            # 2) Si un trigger a été émis sur l'image complète, on utilise cet état
            if triggered:
                final_bbox = bbox
                trigger_source = "QNetwork"
            else:
                # Tentative sur les 4 coins
                final_bbox = None
                trigger_source = None
                for corner in corners:
                    history_deque = deque([torch.zeros(num_actions).to(device) for _ in range(10)], maxlen=10)
                    bbox = list(corner)  # Réinitialisation avec le coin
                    corner_triggered = False
                    for step in range(30):
                        detection_actions += 1
                        total_non_trigger_actions += 1

                        current_iou = calculate_iou(bbox, gt)
                        
                        features = get_features(bbox, cnn_model, image_tensor, device, transform)
                        history_tensor = get_history_tensor(history_deque)
                        with torch.no_grad():
                            q_values = q_network(features, history_tensor)
                        action_idx = torch.argmax(q_values).item()
                        action = ACTIONS[action_idx]
                        
                        if action == "trigger":
                            corner_triggered = True
                            qnetwork_trigger_count += 1
                            trigger_reward = calculate_reward(current_iou, current_iou, action_idx)
                            break
                        else:
                            new_bbox = apply_action(bbox, action, img_w, img_h)
                            new_iou = calculate_iou(new_bbox, gt)
                            total_non_trigger_actions += 1
                            if new_iou > current_iou:
                                improved_actions += 1
                            bbox = new_bbox
                            one_hot_action = torch.zeros(num_actions).to(device)
                            one_hot_action[action_idx] = 1
                            history_deque.append(one_hot_action)
                    if corner_triggered:
                        final_bbox = bbox
                        trigger_source = "QNetwork"
                        break
                # 3) Si aucun trigger n'est obtenu sur aucun coin, on force le trigger
                if final_bbox is None:
                    final_bbox = etat_final_1  # Utilisation de l'état finale 1
                    trigger_source = "forced"
                    forced_trigger_count += 1

            # Calcul de l'IoU final pour cette groundtruth
            final_iou = calculate_iou(final_bbox, gt)
            final_ious.append(final_iou)
            actions_per_detection.append(detection_actions)
            total_triggers += 1  # Un trigger par groundtruth

            # Comptage des triggers réussis (IoU >= IOU_THRESHOLD)
            if final_iou >= IOU_THRESHOLD:
                successful_triggers += 1
                if trigger_source == "QNetwork":
                    qnetwork_trigger_success += 1

            # Dessiner une croix noire sur la groundtruth détectée afin d'éviter de redétecter le même objet
            gx1, gy1, gx2, gy2 = gt["xmin"], gt["ymin"], gt["xmax"], gt["ymax"]
            draw_cross(
                image_pil,
                (gx1, gy1, gx2, gy2),
                color=(0, 0, 0),
                w=int(0.25 * abs(gx1 - gx2)),
                h=int(0.25 * abs(gy1 - gy2))
            )
            # On met à jour l'image (optionnel, si les prochaines détections utilisent l'image modifiée)
            image_tensor = pil_to_tensor(image_pil).to(device)

    # Calcul des métriques finales
    percent_detections_iou = 100 * successful_triggers / total_triggers if total_triggers > 0 else 0.0
    average_final_iou = np.mean(final_ious) if final_ious else 0.0

    # Calcul de la précision, recall et f1_score
    precision = successful_triggers / total_triggers if total_triggers > 0 else 0.0
    recall = successful_triggers / total_groundtruths if total_groundtruths > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    percent_qnetwork_triggers = 100 * qnetwork_trigger_count / total_triggers if total_triggers > 0 else 0.0
    percent_qnetwork_triggers_success = 100 * qnetwork_trigger_success / qnetwork_trigger_count if qnetwork_trigger_count > 0 else 0.0
    percent_actions_improved = 100 * improved_actions / total_non_trigger_actions if total_non_trigger_actions > 0 else 0.0
    avg_actions_per_detection = np.mean(actions_per_detection) if actions_per_detection else 0.0

    metrics = {
        "percent_detections_iou_above_0.6": percent_detections_iou,
        "average_final_iou": average_final_iou,
        "f1_score": f1_score,
        "forced_trigger_count": forced_trigger_count,
        "percent_qnetwork_triggers": percent_qnetwork_triggers,
        "percent_qnetwork_triggers_success": percent_qnetwork_triggers_success,
        "percent_actions_improved": percent_actions_improved,
        "avg_actions_per_detection": avg_actions_per_detection
    }

    return metrics, final_ious, actions_per_detection

# Préparation des résultats
model_path = models_dir

# Initialisation des réseaux
q_network = QNetwork(input_dim, history_dim, num_actions).to(device)
q_network.load_state_dict(torch.load(model_path, map_location=device))
q_network.eval()

cnn_model = cnn.to(device)
cnn_model.eval()

# Exécution du test
metrics, final_ious, actions_per_detection = test(q_network, cnn_model, test_list, images_dir_test, annotations_dir_test)

# Affichage des métriques d'évaluation
print("Results:")
print(f"  % de détections avec IoU final > 0.6 : {metrics['percent_detections_iou_above_0.6']:.2f}%")
print(f"  Moyenne de IoU finale : {metrics['average_final_iou']:.4f}")
print(f"  f1_score (precision & recall) : {metrics['f1_score']:.4f}")
print(f"  Nombre de triggers forcé : {metrics['forced_trigger_count']}")
print(f"  % de triggers initiés par le QNetwork : {metrics['percent_qnetwork_triggers']:.2f}%")
print(f"  % de triggers QNetwork avec IoU > 0.6 : {metrics['percent_qnetwork_triggers_success']:.2f}%")
print(f"  % d'actions du QNetwork ayant amélioré l'IoU : {metrics['percent_actions_improved']:.2f}%")
print(f"  Nombre moyen d'actions par détection : {metrics['avg_actions_per_detection']:.2f}")

# Affichage d'histogrammes pour quelques métriques
plt.figure(figsize=(18, 5))
plt.hist(final_ious, bins=20, color='salmon', edgecolor='black')
plt.title("IoU finale par détection")
plt.xlabel("IoU")
plt.ylabel("Fréquence")
plt.tight_layout()
plt.show()

plt.figure(figsize=(18, 5))
plt.hist(actions_per_detection, bins=20, color='skyblue', edgecolor='black')
plt.title("Nombre d'actions par détection")
plt.xlabel("Nombre d'actions")
plt.ylabel("Fréquence")
plt.tight_layout()
plt.show()
