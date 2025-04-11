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

from model_code.model_DQN_ResNet50 import cnn, transform, QNetwork
from additional_functions import (
    ACTIONS, sort_bounding_boxes_by_area, get_best_action,
    calculate_iou, pil_to_tensor, get_history_tensor, 
    apply_action, calculate_reward, draw_cross,
    load_annotations, get_features
)

# ============================================================================
# 1) Paramètres globaux
# ============================================================================
ROOT = ""
target_type = "dog"

super_dataset_dir = os.path.join(ROOT, "SuperDataset")
images_dir_train = os.path.join(super_dataset_dir, target_type, "imgs")
annotations_dir_train = os.path.join(super_dataset_dir, target_type, "annotations")

train_list = [
    os.path.splitext(f)[0]
    for f in os.listdir(images_dir_train)
    if f.endswith(".jpg")
]

NUM_ACTIONS = 9
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {DEVICE}")

cnn = cnn.to(DEVICE)
cnn.eval()
for param in cnn.parameters():
    param.requires_grad = False

input_dim = 2048
history_dim = 90
num_actions = NUM_ACTIONS

q_network = QNetwork(input_dim, history_dim, NUM_ACTIONS).to(DEVICE)
target_network = QNetwork(input_dim, history_dim, NUM_ACTIONS).to(DEVICE)
target_network.load_state_dict(q_network.state_dict())

REPLAY_MEMORY_SIZE = 500_000
REPLAY_START_SIZE = 50_000
MINIBATCH_SIZE = 128
TARGET_NETWORK_UPDATE_FREQ = 10_000
LEARNING_RATE = 2e-4
NUM_EPOCHS = 100
NUM_STEP_FOR_TRAIN = int(MINIBATCH_SIZE / 8)
print(f"Number of training steps per update: {NUM_STEP_FOR_TRAIN}")

DISCOUNT_FACTOR = 0.99

optimizer = optim.Adam(q_network.parameters(), lr=LEARNING_RATE, weight_decay=1e-6)
loss_fn = nn.MSELoss()

replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
q_network.train()

# -- Fonction d'épsilon (planning d'exploration)
def get_epsilon(epoch):
    if epoch <= 9:
        eps = 1 + (epoch - 1) * (0.1 - 1) / (9 - 1)
    elif epoch <= 40:
        eps = 0.1
    elif epoch <= 100:
        eps = 0.1 + (epoch - 40) * (0.01 - 0.1) / (100 - 40)
    else:
        eps = 0.05
    return eps

# -- Calcul du nombre total d'annotations
total_annotations_count = 0
for image_id in train_list:
    annotation_file = os.path.join(annotations_dir_train, image_id + ".xml")
    ground_truths = load_annotations(annotation_file)
    total_annotations_count += len(ground_truths)

total_steps = 0
update_steps = 0
steps_since_last_train = 0

csv_file_path = os.path.join(ROOT, "training_metrics_DNQ_Perso_2.csv")
if not os.path.exists(csv_file_path):
    with open(csv_file_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "Epoch",
            "ReplayMemoryFill",
            "TotalSteps",
            "%IoU>0.6",
            "AvgFinalIoU",
            "StdFinalIoU",
            "NbForcedTrigger",
            "%TriggersQNet",
            "%QNetTriggersIoU>0.6",
            "%QNetActionsImprovedIoU",
            "AvgActionsPerDetection",
            "StdActionsPerDetection"
        ])

# ============================================================================
# 2) Boucle principale d'entraînement
# ============================================================================
for epoch in range(1, NUM_EPOCHS + 1):
    
    # ------------------------------------------------------------------------
    # DQN : on crée une sous-liste aléatoire de 50% des images
    # ------------------------------------------------------------------------
    half_size = int(0.5 * len(train_list))
    shuffled_train_list = random.sample(train_list, k=half_size)
    
    # Si vous souhaitez avoir une barre de progression
    # correspondant exactement au nombre d'annotations de la sous-liste :
    epoch_annotations_count = 0
    for image_id in shuffled_train_list:
        anno_file = os.path.join(annotations_dir_train, f"{image_id}.xml")
        epoch_annotations_count += len(load_annotations(anno_file))
    
    # ------------------------------------------------------------------------
    # Variables de suivi pour les métriques de fin d’époque
    # ------------------------------------------------------------------------
    epoch_total_detections = 0
    
    # -- Pour le IoU final :
    epoch_sum_final_iou = 0.0
    epoch_sum_final_iou_sq = 0.0  
    
    # -- Pour le nombre d’actions par détection :
    epoch_total_actions = 0
    epoch_sum_actions_sq = 0.0    
    
    epoch_count_final_iou_above_0_6 = 0

    # QNetwork stats
    qnet_total_actions = 0
    qnet_improved_count = 0
    qnet_trigger_count = 0
    qnet_trigger_iou_above_0_6 = 0
    Nb_forced_trigger = 0
    qnet_action_histogram = [0] * NUM_ACTIONS
    q_values_storage = [[] for _ in range(NUM_ACTIONS)]

    eps = get_epsilon(epoch)
    print(f"\nEpoch {epoch}/{NUM_EPOCHS}, epsilon = {eps:.2f}, model = {1 - eps:.2f}")

    # pbar avec le nombre d'annotations correspondant à la *sous-liste*
    pbar = tqdm(total=epoch_annotations_count, desc=f"Epoch {epoch}/{NUM_EPOCHS} - Processing annotations")

    # ------------------------------------------------------------------------
    # 3) Parcours du sous-ensemble (50%) de train_list
    # ------------------------------------------------------------------------
    for image_id in shuffled_train_list:
        image_file = os.path.join(images_dir_train, image_id + ".jpg")
        annotation_file = os.path.join(annotations_dir_train, image_id + ".xml")

        image_pil = Image.open(image_file).convert("RGB")
        image_tensor = pil_to_tensor(image_pil).to(DEVICE)

        ground_truths = load_annotations(annotation_file)
        sorted_ground_truths = sort_bounding_boxes_by_area(ground_truths)

        image_draw = ImageDraw.Draw(image_pil)

        for ground_truth in sorted_ground_truths:
            pbar.update(1)
            
            # BBox initiale = l'image entière
            bbox = [0, 0, image_tensor.shape[2], image_tensor.shape[1]]
            
            # Historique d'actions
            history_deque = deque([
                torch.zeros(NUM_ACTIONS).to(DEVICE) for _ in range(10)
            ], maxlen=10)
            history_tensor = get_history_tensor(history_deque)
            
            current_iou = calculate_iou(bbox, ground_truth)
            prev_iou = current_iou
            done = False
            rewards = []
            action_count = 0

            while not done:
                total_steps += 1
                action_count += 1
                img_height, img_width = image_tensor.shape[1:3]

                features = get_features(bbox, cnn, image_tensor, DEVICE, transform)
                rand_val = random.random()

                # -- Epsilon greedy
                if rand_val < eps:
                    chosen_by_qnet = False
                    action_idx = get_best_action(bbox, ground_truth, img_width, img_height)
                    chosen_q_value = 0.0
                else:
                    chosen_by_qnet = True
                    with torch.no_grad():
                        q_values = q_network(features, history_tensor)
                        action_idx = torch.argmax(q_values).item()
                        chosen_q_value = q_values[0, action_idx].item()

                original_qnet_choice = chosen_by_qnet
                forced_trigger = False

                # -- Force trigger si trop d'itérations
                if action_count > 50:
                    action_idx = 8  # index 8 -> "trigger"
                    forced_trigger = True
                    chosen_by_qnet = False

                action = ACTIONS[action_idx]

                # ------------------------------------------------------------
                # Application de l'action / calcul de la récompense
                # ------------------------------------------------------------
                if action == "trigger":
                    if forced_trigger:
                        Nb_forced_trigger += 1
                    
                    reward = calculate_reward(
                        calculate_iou(bbox, ground_truth), prev_iou, action_idx
                    )
                    rewards.append(reward)
                    current_iou = calculate_iou(bbox, ground_truth)
                    
                    # Marquer la bbox
                    gx1, gy1, gx2, gy2 = (
                        ground_truth["xmin"],
                        ground_truth["ymin"],
                        ground_truth["xmax"],
                        ground_truth["ymax"]
                    )
                    draw_cross(
                        image_pil, (gx1, gy1, gx2, gy2),
                        color=(0, 0, 0),
                        w=int(0.25 * abs(gx1 - gx2)),
                        h=int(0.25 * abs(gy1 - gy2))
                    )
                    image_tensor = pil_to_tensor(image_pil).to(DEVICE)
                    done = True
                else:
                    bbox = apply_action(bbox, action, img_width, img_height)
                    current_iou = calculate_iou(bbox, ground_truth)
                    reward = calculate_reward(current_iou, prev_iou, action_idx)
                    rewards.append(reward)
                    
                    if original_qnet_choice and (not forced_trigger) and current_iou > prev_iou:
                        qnet_improved_count += 1
                    prev_iou = current_iou

                # -- Mise à jour de l'historique
                one_hot_action = torch.zeros(NUM_ACTIONS).to(DEVICE)
                one_hot_action[action_idx] = 1
                history_deque.append(one_hot_action)
                history_tensor = get_history_tensor(history_deque)

                # -- Mémorisation (replay)
                replay_memory.append((
                    features.squeeze(0).detach().cpu(),   
                    history_tensor.squeeze(0).cpu(),      
                    action_idx,
                    rewards[-1],
                    done
                ))



                # Comptabilise si l'action venait du QNetwork
                if original_qnet_choice and not forced_trigger:
                    qnet_total_actions += 1
                    qnet_action_histogram[action_idx] += 1
                    q_values_storage[action_idx].append(chosen_q_value)
                    if action == "trigger":
                        qnet_trigger_count += 1
                        if current_iou > 0.6:
                            qnet_trigger_iou_above_0_6 += 1

                # -- Entraînement du réseau sur un batch
                steps_since_last_train += 1
                if (
                    len(replay_memory) >= REPLAY_START_SIZE
                    and steps_since_last_train > NUM_STEP_FOR_TRAIN
                ):
                    steps_since_last_train = 0
                    batch = random.sample(replay_memory, MINIBATCH_SIZE)
                    
                    (batch_states, batch_histories,
                     batch_actions, batch_rewards, batch_dones) = zip(*batch)
                    
                    batch_states = torch.stack(batch_states).to(DEVICE)
                    batch_histories = torch.stack(batch_histories).to(DEVICE)
                    batch_actions = torch.tensor(batch_actions).to(DEVICE)
                    batch_rewards = torch.tensor(batch_rewards).to(DEVICE)
                    batch_dones = torch.tensor(batch_dones).to(DEVICE)

                    # -- DoubleDQN
                    with torch.no_grad():
                        next_q_values_online = q_network(batch_states, batch_histories)
                        best_next_actions = next_q_values_online.argmax(dim=1)

                        next_q_values_target = target_network(batch_states, batch_histories)
                        chosen_next_q_values = next_q_values_target.gather(
                            1, best_next_actions.unsqueeze(1)
                        ).squeeze(1)

                        targets = batch_rewards + (1 - batch_dones.float()) * DISCOUNT_FACTOR * chosen_next_q_values

                    current_q_values = q_network(batch_states, batch_histories)
                    current_q_values = current_q_values.gather(
                        1, batch_actions.unsqueeze(1)
                    ).squeeze()

                    loss = loss_fn(current_q_values, targets).mean()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    update_steps += 1
                    if update_steps % TARGET_NETWORK_UPDATE_FREQ == 0:
                        target_network.load_state_dict(q_network.state_dict())

            # -- Fin du while (bbox terminée)
            epoch_total_detections += 1
            epoch_total_actions += action_count
            epoch_sum_actions_sq += (action_count ** 2)
            
            epoch_sum_final_iou += current_iou
            epoch_sum_final_iou_sq += (current_iou ** 2)
            if current_iou > 0.6:
                epoch_count_final_iou_above_0_6 += 1

    pbar.close()

    # ------------------------------------------------------------------------
    # 4) Calcul des métriques de fin d'époque
    # ------------------------------------------------------------------------
    if epoch_total_detections > 0:
        # IoU final
        avg_final_iou = epoch_sum_final_iou / epoch_total_detections
        mean_iou_sq = epoch_sum_final_iou_sq / epoch_total_detections
        var_final_iou = mean_iou_sq - (avg_final_iou ** 2)
        std_final_iou = np.sqrt(var_final_iou) if var_final_iou > 0 else 0.0

        # Actions par détection
        avg_actions_per_detection = epoch_total_actions / epoch_total_detections
        mean_actions_sq = epoch_sum_actions_sq / epoch_total_detections
        var_actions = mean_actions_sq - (avg_actions_per_detection ** 2)
        std_actions = np.sqrt(var_actions) if var_actions > 0 else 0.0

        perc_final_iou_above_0_6 = (
            epoch_count_final_iou_above_0_6 / epoch_total_detections * 100
        )
    else:
        avg_final_iou = 0.0
        std_final_iou = 0.0
        avg_actions_per_detection = 0.0
        std_actions = 0.0
        perc_final_iou_above_0_6 = 0.0

    # QNetwork metrics
    perc_qnet_improved = (
        qnet_improved_count / qnet_total_actions * 100
        if qnet_total_actions > 0 else 0
    )
    perc_qnet_trigger_iou = (
        qnet_trigger_iou_above_0_6 / qnet_trigger_count * 100
        if qnet_trigger_count > 0 else 0
    )
    perc_qnet_trigger = (
        qnet_trigger_count / epoch_total_detections * 100
        if epoch_total_detections > 0 else 0
    )

    replay_fill = len(replay_memory)
    print(f"Replay memory: {replay_fill} / {REPLAY_MEMORY_SIZE} ({(replay_fill / REPLAY_MEMORY_SIZE) * 100:.2f}%)")
    print(f"Total steps since beginning: {total_steps}\n")

    print("Histogramme des actions choisies par le QNetwork :")
    if qnet_total_actions > 0:
        max_count = max(qnet_action_histogram)
        for i in range(NUM_ACTIONS):
            count = qnet_action_histogram[i]
            bar_length = int((count / max_count) * 50) if max_count > 0 else 0
            bar = "█" * bar_length
            perc = (count / qnet_total_actions) * 100
            if len(q_values_storage[i]) > 0:
                avg_q = np.mean(q_values_storage[i])
                var_q = np.var(q_values_storage[i])
                std_q = np.sqrt(var_q)
            else:
                avg_q, std_q = 0.0, 0.0
            print(
                f"  Action '{ACTIONS[i]}': {count:4d} fois  {bar:<50} {perc:5.1f}%  "
                f"Moyenne Q: {avg_q:.4f}, Ecart-type Q: {std_q:.4f}"
            )
    else:
        print("Aucune action QNetwork enregistrée.")
    print("")

    print(f"\n==== Metrics for Epoch {epoch} ====")
    print(f"% de détections avec IoU final > 0.6       : {perc_final_iou_above_0_6:.2f}%")
    print(f"Moyenne de IoU finale (±ecart-type)        : {avg_final_iou:.4f} ± {std_final_iou:.4f}")
    print(f"Nombre de triggers forcé                   : {Nb_forced_trigger}")
    print(f"% de triggers QNetwork                     : {perc_qnet_trigger:.2f}%")
    print(f"% de triggers QNet IoU>0.6                : {perc_qnet_trigger_iou:.2f}%")
    print(f"% d'actions QNet améliorant IoU            : {perc_qnet_improved:.2f}%")
    print(f"Nombre moyen d'actions/détection (±ecart)  : {avg_actions_per_detection:.2f} ± {std_actions:.2f}")

    # ------------------------------------------------------------------------
    # 5) Enregistrement des métriques dans le CSV
    # ------------------------------------------------------------------------
    with open(csv_file_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            epoch,
            replay_fill,
            total_steps,
            f"{perc_final_iou_above_0_6:.2f}",
            f"{avg_final_iou:.4f}",
            f"{std_final_iou:.4f}",
            f"{Nb_forced_trigger}",
            f"{perc_qnet_trigger:.2f}",
            f"{perc_qnet_trigger_iou:.2f}",
            f"{perc_qnet_improved:.2f}",
            f"{avg_actions_per_detection:.2f}",
            f"{std_actions:.2f}"
        ])

    # -- Sauvegarde du modèle
    if epoch > 10:
        SAVE_PATH = os.path.join(ROOT, f"../models_saves/DQN/ResNet50/model_epoch_{epoch:02d}_DoubleDQN_Perso_2.pth")
        torch.save(q_network.state_dict(), SAVE_PATH)
        print(f"Model saved at epoch {epoch} to {SAVE_PATH}")
