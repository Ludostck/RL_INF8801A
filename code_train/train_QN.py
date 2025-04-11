import csv
import os
import random
from collections import deque
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from PIL import Image, ImageDraw

# Importation des modules du modèle et des fonctions supplémentaires
from model_code.model_QN_ResNet50 import cnn, transform, QNetwork
from additional_functions import (
    ACTIONS, sort_bounding_boxes_by_area, get_best_action,
    calculate_iou, pil_to_tensor, get_history_tensor, 
    apply_action, calculate_reward, draw_cross,
    load_annotations, get_features
)

# ============================================================================
# 1) Paramètres globaux et initialisation
# ============================================================================

ROOT = ""
target_type = "dog"

# Chemins vers les répertoires d'images et d'annotations
super_dataset_dir = os.path.join(ROOT, "SuperDataset")
images_dir_train = os.path.join(super_dataset_dir, target_type, "imgs")
annotations_dir_train = os.path.join(super_dataset_dir, target_type, "annotations")

# Création de la liste des identifiants d'images (sans extension)
train_list = [os.path.splitext(f)[0] for f in os.listdir(images_dir_train) if f.endswith(".jpg")]

NUM_ACTIONS = 9
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")

# Initialisation du CNN (pour l'extraction de caractéristiques)
cnn = cnn.to(device)
cnn.eval()
for param in cnn.parameters():
    param.requires_grad = False

# Dimensions d'entrée pour le QNetwork
input_dim = 2048
history_dim = 90
num_actions = NUM_ACTIONS

# Initialisation du QNetwork et du réseau cible
q_network = QNetwork(input_dim, history_dim, num_actions).to(device)
target_network = QNetwork(input_dim, history_dim, num_actions).to(device)
target_network.load_state_dict(q_network.state_dict())
target_network.eval()

# Paramètres de l'expérience et de l'apprentissage
REPLAY_MEMORY_SIZE = 500_000
REPLAY_START_SIZE = 100_000
MINIBATCH_SIZE = 128
TARGET_NETWORK_UPDATE_FREQ = 10_000
LEARNING_RATE = 2e-4
NUM_EPOCHS = 100
NUM_STEP_FOR_TRAIN = int(MINIBATCH_SIZE / 8)
print(f"Number of training steps per update: {NUM_STEP_FOR_TRAIN}")

border_size = 16
DISCOUNT_FACTOR = 0.99

optimizer = optim.Adam(q_network.parameters(), lr=LEARNING_RATE, weight_decay=1e-6)

# ============================================================================
# 2) Fonction d'épsilon (pour le planning de l'exploration)
# ============================================================================

def get_epsilon(epoch):
    """
    Calcule la valeur d'épsilon en fonction de l'époque pour la stratégie epsilon-greedy.
    """
    if epoch <= 6:
        eps = 1 + (epoch - 1) * (0.1 - 1) / (6 - 1)
    elif epoch <= 15:
        eps = 0.1
    elif epoch <= 30:
        eps = 0.1 + (epoch - 16) * (0.01 - 0.1) / (25 - 16)
    else:
        eps = 0.01
    return eps

loss_fn = nn.MSELoss()
replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

q_network.train()

# Comptage du nombre total d'annotations pour toutes les images
total_annotations_count = 0
for image_id in train_list:
    annotation_file = os.path.join(annotations_dir_train, image_id + ".xml")
    ground_truths = load_annotations(annotation_file)
    total_annotations_count += len(ground_truths)

total_steps = 0
update_steps = 0
steps_since_last_train = 0

# ============================================================================
# 3) Préparation du fichier CSV pour la journalisation des métriques
# ============================================================================

csv_file_path = os.path.join(ROOT, "training_metrics_Base_Perso_2.csv")
# Si le fichier n'existe pas, on le crée et on y écrit l'en-tête
if not os.path.exists(csv_file_path):
    with open(csv_file_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "Epoch",
            "ReplayMemoryFill",
            "TotalSteps",
            "%IoU>0.6",
            "AvgFinalIoU",
            "NbForcedTrigger",
            "%TriggersQNet",
            "%QNetTriggersIoU>0.6",
            "%QNetActionsImprovedIoU",
            "AvgActionsPerDetection"
        ])

# ============================================================================
# 4) Boucle principale d'entraînement
# ============================================================================

for epoch in range(1, NUM_EPOCHS + 1):
    
    # Initialisation des compteurs et accumulateurs pour l'époque
    epoch_total_detections = 0
    epoch_total_actions = 0
    epoch_sum_final_iou = 0.0
    epoch_count_final_iou_above_0_6 = 0

    # Statistiques liées aux actions choisies par le QNetwork
    qnet_total_actions = 0
    qnet_improved_count = 0
    qnet_trigger_count = 0
    qnet_trigger_iou_above_0_6 = 0
    Nb_forced_trigger = 0
    qnet_action_histogram = [0] * NUM_ACTIONS
    q_values_storage = [[] for _ in range(NUM_ACTIONS)]
    
    eps = get_epsilon(epoch)
    print(f"\nEpoch {epoch}/{NUM_EPOCHS}, epsilon = {eps:.2f}, model = {1 - eps:.2f}")
    pbar = tqdm(total=total_annotations_count, desc=f"Epoch {epoch}/{NUM_EPOCHS} - Processing annotations")

    # Parcours de chaque image de la liste d'entraînement
    for image_id in train_list:
        image_file = os.path.join(images_dir_train, image_id + ".jpg")
        annotation_file = os.path.join(annotations_dir_train, image_id + ".xml")

        # Chargement de l'image et conversion en tenseur
        image_pil = Image.open(image_file).convert("RGB")
        image_tensor = pil_to_tensor(image_pil).to(device)

        # Chargement et tri des annotations de vérité terrain par aire croissante
        ground_truths = load_annotations(annotation_file)
        sorted_ground_truths = sort_bounding_boxes_by_area(ground_truths)

        # Préparation de l'image pour l'affichage (pour dessiner les résultats)
        image_draw = ImageDraw.Draw(image_pil)

        # Pour chaque boîte de vérité terrain de l'image
        for ground_truth in sorted_ground_truths:
            pbar.update(1)
            
            # Initialisation de la boîte englobante (l'image entière)
            bbox = [0, 0, image_tensor.shape[2], image_tensor.shape[1]]
            
            # Initialisation de l'historique des actions sous forme de deque
            history_deque = deque([torch.zeros(NUM_ACTIONS).to(device) for _ in range(10)], maxlen=10)
            history_tensor = get_history_tensor(history_deque)
            
            # Calcul de l'IoU initiale entre la bbox et la vérité terrain
            current_iou = calculate_iou(bbox, ground_truth)
            prev_iou = current_iou
            done = False
            rewards = []
            action_count = 0

            # Boucle d'exécution des actions jusqu'à ce qu'une action "trigger" termine la détection
            while not done:
                total_steps += 1
                action_count += 1
                img_height, img_width = image_tensor.shape[1:3]

                # Extraction des caractéristiques de la région définie par la bbox
                features = get_features(bbox, cnn, image_tensor, device, transform)
                rand_val = random.random()

                # --- Stratégie epsilon-greedy ---
                if rand_val < eps:
                    chosen_by_qnet = False
                    action_idx = get_best_action(bbox, ground_truth, img_width, img_height)
                else:
                    chosen_by_qnet = True
                    with torch.no_grad():
                        q_values = q_network(features, history_tensor)
                        action_idx = torch.argmax(q_values).item()
                        chosen_q_value = q_values[0, action_idx].item()

                original_qnet_choice = chosen_by_qnet
                forced_trigger = False

                # Force le trigger si le nombre d'actions dépasse 50 (pour éviter des boucles infinies)
                if action_count > 50:
                    action_idx = 8  # index 8 correspond à "trigger"
                    forced_trigger = True
                    chosen_by_qnet = False

                action = ACTIONS[action_idx]

                # --- Application de l'action et calcul de la récompense ---
                if action == "trigger":
                    if forced_trigger:
                        Nb_forced_trigger += 1
                    
                    reward = calculate_reward(calculate_iou(bbox, ground_truth), prev_iou, action_idx)
                    rewards.append(reward)
                    current_iou = calculate_iou(bbox, ground_truth)
                    
                    # Dessin d'une croix sur l'image pour marquer la bbox de la vérité terrain
                    gx1, gy1, gx2, gy2 = ground_truth["xmin"], ground_truth["ymin"], ground_truth["xmax"], ground_truth["ymax"]
                    draw_cross(
                        image_pil, (gx1, gy1, gx2, gy2), color=(0, 0, 0),
                        w=int(0.25 * abs(gx1 - gx2)), h=int(0.25 * abs(gy1 - gy2))
                    )
                    image_tensor = pil_to_tensor(image_pil).to(device)
                    done = True
                else:
                    bbox = apply_action(bbox, action, img_width, img_height)
                    current_iou = calculate_iou(bbox, ground_truth)
                    reward = calculate_reward(current_iou, prev_iou, action_idx)
                    rewards.append(reward)
                    if original_qnet_choice and (not forced_trigger) and current_iou > prev_iou:
                        qnet_improved_count += 1
                    prev_iou = current_iou

                # Mise à jour de l'historique des actions
                one_hot_action = torch.zeros(NUM_ACTIONS).to(device)
                one_hot_action[action_idx] = 1
                history_deque.append(one_hot_action)
                history_tensor = get_history_tensor(history_deque)

                # Enregistrement de la transition dans la mémoire de replay
                replay_memory.append((
                    features.detach().cpu(),
                    history_tensor.squeeze(0).cpu(),
                    action_idx,
                    rewards[-1],
                    done
                ))

                # Enregistrement des statistiques si l'action provient du QNetwork
                if original_qnet_choice and not forced_trigger:
                    qnet_total_actions += 1
                    qnet_action_histogram[action_idx] += 1
                    q_values_storage[action_idx].append(chosen_q_value)
                    if action == "trigger":
                        qnet_trigger_count += 1
                        if current_iou > 0.6:
                            qnet_trigger_iou_above_0_6 += 1

                # Entraînement sur un batch si les conditions sont réunies
                steps_since_last_train += 1
                if len(replay_memory) >= REPLAY_START_SIZE and steps_since_last_train > NUM_STEP_FOR_TRAIN:
                    steps_since_last_train = 0
                    batch = random.sample(replay_memory, MINIBATCH_SIZE)
                    batch_states, batch_histories, batch_actions, batch_rewards, batch_dones = zip(*batch)
                    batch_states = torch.stack(batch_states).to(device)
                    batch_histories = torch.stack(batch_histories).to(device)
                    batch_actions = torch.tensor(batch_actions).to(device)
                    batch_rewards = torch.tensor(batch_rewards).to(device)
                    batch_dones = torch.tensor(batch_dones).to(device)
                    batch_states = batch_states.view(batch_states.size(0), -1)

                    with torch.no_grad():
                        next_q_values = target_network(batch_states, batch_histories)
                        max_next_q_values = next_q_values.max(dim=1)[0]
                        targets = batch_rewards + (1 - batch_dones.float()) * DISCOUNT_FACTOR * max_next_q_values

                    current_q_values = q_network(batch_states, batch_histories).gather(1, batch_actions.unsqueeze(1)).squeeze()
                    loss = loss_fn(current_q_values, targets).mean()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    update_steps += 1
                    if update_steps % TARGET_NETWORK_UPDATE_FREQ == 0:
                        target_network.load_state_dict(q_network.state_dict())

            # Fin du traitement de la détection pour la boîte courante
            epoch_total_detections += 1
            epoch_total_actions += action_count
            epoch_sum_final_iou += current_iou
            if current_iou > 0.6:
                epoch_count_final_iou_above_0_6 += 1

    pbar.close()

    # ========================================================================
    # 5) Calcul et affichage des métriques de fin d'époque
    # ========================================================================
    avg_final_iou = epoch_sum_final_iou / epoch_total_detections if epoch_total_detections > 0 else 0
    perc_final_iou_above_0_6 = (epoch_count_final_iou_above_0_6 / epoch_total_detections * 100) if epoch_total_detections > 0 else 0
    avg_actions_per_detection = epoch_total_actions / epoch_total_detections if epoch_total_detections > 0 else 0
    perc_qnet_improved = (qnet_improved_count / qnet_total_actions * 100) if qnet_total_actions > 0 else 0
    perc_qnet_trigger_iou = (qnet_trigger_iou_above_0_6 / qnet_trigger_count * 100) if qnet_trigger_count > 0 else 0
    perc_qnet_trigger = (qnet_trigger_count / epoch_total_detections * 100) if epoch_total_detections > 0 else 0

    replay_fill = len(replay_memory)
    print(f"Replay memory: {replay_fill} / {REPLAY_MEMORY_SIZE} ({(replay_fill / REPLAY_MEMORY_SIZE) * 100:.2f}%)")
    print(f"Total steps since beginning: {total_steps}\n")

    # Affichage de l'histogramme des actions choisies par le QNetwork
    print("Histogramme des actions choisies par le QNetwork :")
    if qnet_total_actions > 0:
        max_count = max(qnet_action_histogram)
        for i in range(NUM_ACTIONS):
            count = qnet_action_histogram[i]
            bar_length = int((count / max_count) * 50) if max_count > 0 else 0
            bar = "█" * bar_length
            perc = (count / qnet_total_actions) * 100
            if q_values_storage[i]:
                avg_q = np.mean(q_values_storage[i])
                var_q = np.var(q_values_storage[i])
            else:
                avg_q, var_q = 0.0, 0.0
            print(f"  Action '{ACTIONS[i]}': {count:4d} fois  {bar:<50} {perc:5.1f}%  "
                  f"Moyenne Q: {avg_q:.4f}, Variance Q: {var_q:.4f}")
    else:
        print("Aucune action QNetwork enregistrée.")
    print("")
    print("\n==== Metrics for Epoch {} ====".format(epoch))
    print(f"% de détections avec IoU final > 0.6 : {perc_final_iou_above_0_6:.2f}%")
    print(f"Moyenne de IoU finale : {avg_final_iou:.4f}")
    print(f"Nombre de triggers forcé : {Nb_forced_trigger:.2f}")
    print(f"% de triggers initiés par le QNetwork : {perc_qnet_trigger:.2f}%")
    print(f"% de triggers initiés par le QNetwork avec IoU > 0.6 : {perc_qnet_trigger_iou:.2f}%")
    print(f"% d'actions du QNetwork ayant amélioré l'IoU : {perc_qnet_improved:.2f}%")
    print(f"Nombre moyen d'actions par détection : {avg_actions_per_detection:.2f}")

    # ========================================================================
    # 6) Enregistrement des métriques de fin d'époque dans le fichier CSV
    # ========================================================================
    with open(csv_file_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            epoch,
            replay_fill,                        # Taille actuelle de la mémoire de replay
            total_steps,                        # Nombre total d'étapes réalisées
            f"{perc_final_iou_above_0_6:.2f}",  # Pourcentage d'IoU > 0.6
            f"{avg_final_iou:.4f}",             # IoU finale moyenne
            f"{Nb_forced_trigger}",             # Nombre de triggers forcés
            f"{perc_qnet_trigger:.2f}",         # Pourcentage de triggers initiés par le QNetwork
            f"{perc_qnet_trigger_iou:.2f}",     # Pourcentage de triggers QNet avec IoU > 0.6
            f"{perc_qnet_improved:.2f}",        # Pourcentage d'actions du QNetwork améliorant l'IoU
            f"{avg_actions_per_detection:.2f}" # Nombre moyen d'actions par détection
        ])

    # ========================================================================
    # 7) Sauvegarde optionnelle du modèle entraîné
    # ========================================================================
    if epoch > 10:
        SAVE_PATH = os.path.join(ROOT, f"../models_saves/QN/ResNet50/model_epoch_{epoch:02d}_{target_type}_BASE_Perso_2.pth")
        torch.save(q_network.state_dict(), SAVE_PATH)
        print(f"Model saved at epoch {epoch} to {SAVE_PATH}")
        print(f"Loaded model path: {SAVE_PATH}")
