import os
import re
import torch
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from collections import deque
from model_Base_Perso_2 import QNetwork, cnn, transform
from additional_functions import (
    ACTIONS, pil_to_tensor, calculate_iou, apply_action, calculate_reward,
    get_history_tensor, draw_cross, load_annotations, get_features
)
from tqdm import tqdm

# Définition du device (GPU si disponible, sinon CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Testing on device:", device)

# Définition des chemins
ROOT = ""  # Modifier si besoin
super_dataset_dir = os.path.join(ROOT, "TEST")
target_type = "dog"
images_dir_test = os.path.join(super_dataset_dir, target_type, "imgs")
annotations_dir_test = os.path.join(super_dataset_dir, target_type, "annotations")

# Liste des identifiants d'images (sans extensions)
test_list = [os.path.splitext(f)[0] for f in os.listdir(images_dir_test) if f.endswith(".jpg")]

# Paramètres du modèle
input_dim =2048
history_dim = 90
num_actions = 9

IOU_THRESHOLD = 0.6

def test(q_network, cnn_model, test_list, images_dir_test, annotations_dir_test):
    """
    Applique le protocole de test sur le jeu de test.
    Renvoie un dictionnaire de métriques ainsi que les listes d'IoU finales et
    du nombre d'actions par détection.
    """
    final_ious = []
    actions_per_detection = []
    
    total_triggers = 0         # Un trigger par groundtruth
    successful_triggers = 0    # Détections avec IoU >= IOU_THRESHOLD
    forced_trigger_count = 0   # Nombre de triggers forcés
    qnetwork_trigger_count = 0 # Nombre de triggers initiés par le QNetwork
    qnetwork_trigger_success = 0  # Parmi les triggers QNetwork, ceux avec IoU >= IOU_THRESHOLD

    total_non_trigger_actions = 0
    improved_actions = 0
    total_groundtruths = 0

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
            detection_actions = 0
            history_deque = deque([torch.zeros(num_actions).to(device) for _ in range(10)], maxlen=10)

            # 1) Tentative sur l'image complète
            bbox_full = [0, 0, img_w, img_h]
            bbox = bbox_full.copy()
            triggered = False
            trigger_reward = 0.0
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

            # 2) Si un trigger a été émis sur l'image complète, on l'utilise
            if triggered:
                final_bbox = bbox
                trigger_source = "QNetwork"
            else:
                # Tentative sur les 4 coins
                final_bbox = None
                trigger_source = None
                for corner in corners:
                    history_deque = deque([torch.zeros(num_actions).to(device) for _ in range(10)], maxlen=10)
                    bbox = list(corner)
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
                # 3) Si aucun trigger n'est obtenu, forcer le trigger
                if final_bbox is None:
                    final_bbox = etat_final_1
                    trigger_source = "forced"
                    forced_trigger_count += 1

            # Calcul de l'IoU final pour cette groundtruth
            final_iou = calculate_iou(final_bbox, gt)
            final_ious.append(final_iou)
            actions_per_detection.append(detection_actions)
            total_triggers += 1

            if final_iou >= IOU_THRESHOLD:
                successful_triggers += 1
                if trigger_source == "QNetwork":
                    qnetwork_trigger_success += 1

            # Dessin d'une croix noire sur la groundtruth détectée
            gx1, gy1, gx2, gy2 = gt["xmin"], gt["ymin"], gt["xmax"], gt["ymax"]
            draw_cross(
                image_pil,
                (gx1, gy1, gx2, gy2),
                color=(0, 0, 0),
                w=int(0.2 * abs(gx1 - gx2)),
                h=int(0.2 * abs(gy1 - gy2))
            )
            image_tensor = pil_to_tensor(image_pil).to(device)

    percent_detections_iou = 100 * successful_triggers / total_triggers if total_triggers > 0 else 0.0
    average_final_iou = np.mean(final_ious) if final_ious else 0.0

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

def extract_epoch(filename):
    """
    Extrait le numéro d'epoch à partir d'un nom de fichier du type
    "model_epoch_nb_....pth" en utilisant une expression régulière.
    """
    match = re.search(r'model_epoch_(\d+)', filename)
    if match:
        return int(match.group(1))
    return float('inf')

if __name__ == '__main__':
    # Initialisation du modèle CNN (commun à tous les tests)
    cnn_model = cnn.to(device)
    cnn_model.eval()

    # Récupération des modèles depuis le dossier
    models_folder = os.path.join(ROOT, "models/Base/Perso_Resnet50")
    model_files = [f for f in os.listdir(models_folder) if f.endswith(".pth") and "model_epoch_" in f]
    model_files = sorted(model_files, key=extract_epoch)  # Tri dans l'ordre croissant des epochs
    
    all_results = []
    
    for model_file in model_files:
        model_path = os.path.join(models_folder, model_file)
        print(f"\nTesting model: {model_file}")
        
        # Chargement du modèle QNetwork
        q_network = QNetwork(input_dim, history_dim, num_actions).to(device)
        q_network.load_state_dict(torch.load(model_path, map_location=device))
        q_network.eval()
        
        # Exécution du test
        metrics, final_ious, actions_per_detection = test(q_network, cnn_model, test_list, images_dir_test, annotations_dir_test)
        
        epoch = extract_epoch(model_file)
        metrics["model_epoch"] = epoch
        metrics["model_file"] = model_file
        all_results.append(metrics)
        
        # Affichage des métriques pour ce modèle
        print("Results:")
        print(f"  % de détections avec IoU final > 0.6 : {metrics['percent_detections_iou_above_0.6']:.2f}%")
        print(f"  Moyenne de IoU finale : {metrics['average_final_iou']:.4f}")
        print(f"  f1_score (precision & recall) : {metrics['f1_score']:.4f}")
        print(f"  Nombre de triggers forcé : {metrics['forced_trigger_count']}")
        print(f"  % de triggers initiés par le QNetwork : {metrics['percent_qnetwork_triggers']:.2f}%")
        print(f"  % de triggers QNetwork avec IoU > 0.6 : {metrics['percent_qnetwork_triggers_success']:.2f}%")
        print(f"  % d'actions du QNetwork ayant amélioré l'IoU : {metrics['percent_actions_improved']:.2f}%")
        print(f"  Nombre moyen d'actions par détection : {metrics['avg_actions_per_detection']:.2f}")
    
    # Sauvegarde des métriques dans un fichier CSV
    df_results = pd.DataFrame(all_results)
    df_results = df_results.sort_values(by="model_epoch")
    csv_path = os.path.join(ROOT, "results_test_Base_Perso_2.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\nTous les résultats ont été sauvegardés dans {csv_path}")
    
    # Exemple d'affichage de courbes pour quelques métriques selon l'epoch
    plt.figure(figsize=(10, 6))
    plt.plot(df_results["model_epoch"], df_results["average_final_iou"], marker='o')
    plt.title("Moyenne de IoU finale par epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Average Final IoU")
    plt.grid(True)
    plt.show()
    
    plt.figure(figsize=(10, 6))
    plt.plot(df_results["model_epoch"], df_results["f1_score"], marker='o')
    plt.title("f1_score par epoch")
    plt.xlabel("Epoch")
    plt.ylabel("f1_score")
    plt.grid(True)
    plt.show()
