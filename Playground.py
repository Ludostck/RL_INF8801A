import os
import torch
import random
import numpy as np
from collections import deque
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import io
import argparse

# Import des modules du modèle et des fonctions auxiliaires
from code_train.model_code.model_DQN_ResNet50 import cnn, transform, QNetwork
from additional_functions import (
    ACTIONS, sort_bounding_boxes_by_area,
    calculate_iou, pil_to_tensor, get_history_tensor, 
    apply_action,
    load_annotations, get_features
)

def run_inference(image_path, annotation_path, model_checkpoint_path, output_gif_detection, output_gif_iou):
    """
    Pour une image donnée et son fichier d'annotation (XML),
    le script charge le modèle DDQN sauvegardé, et lance la procédure de détection.
    À chaque action, une frame (l'image avec la boîte courante) est sauvegardée.
    Lorsqu'un trigger est atteint, la frame finale est dessinée (boîte finale en vert + boîte GT en bleu).
    L'évolution de l'IoU au cours des actions est également enregistrée pour générer un second GIF.
    Si l'image contient plusieurs objets (plusieurs bbox GT), la détection est répétée pour chacun.
    """
    # Définition du device et mise en place des réseaux
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Charger et préparer le CNN (pour l'extraction des features)
    cnn.to(device)
    cnn.eval()
    
    # Initialiser la QNetwork et charger les poids sauvegardés
    num_actions = len(ACTIONS)
    input_dim = 2048
    history_dim = 90
    q_network = QNetwork(input_dim, history_dim, num_actions).to(device)
    q_network.load_state_dict(torch.load(model_checkpoint_path, map_location=device))
    q_network.eval()
    
    # Charger l'image et le fichier d'annotation
    image_pil = Image.open(image_path).convert("RGB")
    image_tensor = pil_to_tensor(image_pil).to(device)
    ground_truths = load_annotations(annotation_path)
    # Pour le cas de plusieurs objets, on peut trier par aire (ou toute autre stratégie)
    ground_truths = sort_bounding_boxes_by_area(ground_truths)
    
    width, height = image_pil.size

    # Listes pour sauvegarder les frames du GIF de détection et celles du GIF d'évolution IoU
    detection_frames = []
    iou_evolution_frames = []

    # Pour suivre l'évolution globale de l'IoU (on pourra aussi concaténer les évolutions de détection)
    overall_iou_evolution = []
    
    # Pour numéroter l'ensemble des actions
    global_step_counter = 0

    # Pour chaque objet (bbox de vérité terrain) présent dans l'annotation
    for obj_idx, gt in enumerate(ground_truths):
        # Réinitialiser la boîte de départ : toute l'image
        current_bbox = [0, 0, width, height]
        # Initialiser un historique (deque) de 10 actions (vecteurs one-hot à zéro)
        history_deque = deque([torch.zeros(num_actions).to(device) for _ in range(10)], maxlen=10)
        history_tensor = get_history_tensor(history_deque)

        detection_done = False
        detection_step = 0
        local_iou_evolution = []  # Pour stocker l'évolution locale de l'IoU

        # Boucle de détection pour l'objet courant
        while not detection_done:
            detection_step += 1
            global_step_counter += 1
            
            # Extraire les features correspondant à la bbox courante
            features = get_features(current_bbox, cnn, image_tensor, device, transform)
            
            # Mode inference : choix d'action par QNetwork en mode greedy (sans exploration)
            with torch.no_grad():
                q_vals = q_network(features, history_tensor)
                action_idx = torch.argmax(q_vals).item()
            action = ACTIONS[action_idx]
            
            # Sécurité : si trop d'actions, forcer le trigger (ici l'action "trigger" est attendue)
            if detection_step > 50:
                action_idx = ACTIONS.index("trigger")
                action = "trigger"
            
            # Calculer l'IoU entre la bbox courante et la GT de l'objet
            current_iou = calculate_iou(current_bbox, gt)
            local_iou_evolution.append(current_iou)
            overall_iou_evolution.append(current_iou)
            
            # Créer une copie de l'image pour dessiner la bbox courante
            frame = image_pil.copy()
            draw = ImageDraw.Draw(frame)
            # On dessine la bbox courante en rouge
            draw.rectangle(current_bbox, outline="red", width=3)
            # (Optionnel) On peut ajouter le numéro de step et l'IoU en texte :
            # draw.text((5, 5), f"Step: {detection_step}, IoU: {current_iou:.2f}", fill="red")
            detection_frames.append(frame)
            
            # Vérifier si l'action est "trigger" (fin de détection pour cet objet)
            if action == "trigger":
                detection_done = True
                # Créer une frame finale avec la bbox finale (en vert) et la bbox GT (en bleu)
                final_frame = image_pil.copy()
                draw_final = ImageDraw.Draw(final_frame)
                # Dessin de la détection finale
                draw_final.rectangle(current_bbox, outline="green", width=3)
                # Dessin de la vérité terrain en bleu
                gt_bbox = [gt["xmin"], gt["ymin"], gt["xmax"], gt["ymax"]]
                draw_final.rectangle(gt_bbox, outline="blue", width=3)
                # Vous pouvez aussi utiliser draw_cross pour marquer le centre de la bbox GT si souhaité
                detection_frames.append(final_frame)
                # Pour rester longtemps sur la frame finale, on la répète (ici 10 fois)
                for _ in range(10):
                    detection_frames.append(final_frame)
            else:
                # Sinon, appliquer l'action pour mettre à jour la bbox
                current_bbox = apply_action(current_bbox, action, width, height)
            
            # Mettre à jour l'historique des actions avec le one-hot de l'action choisie
            one_hot = torch.zeros(num_actions).to(device)
            one_hot[action_idx] = 1.0
            history_deque.append(one_hot)
            history_tensor = get_history_tensor(history_deque)
        
        # Pour l'objet courant, générer des frames illustrant l'évolution de l'IoU
        for i in range(len(local_iou_evolution)):
            plt.figure(figsize=(4, 3))
            plt.plot(local_iou_evolution[:i+1], marker="o", linestyle="-")
            plt.title(f"Obj {obj_idx+1} - Étape {i+1}")
            plt.xlabel("Étape")
            plt.ylabel("IoU")
            plt.ylim(0, 1)
            plt.grid(True)
            # Enregistrer le graphique dans un buffer mémoire
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plot_img = Image.open(buf).convert("RGB")
            iou_evolution_frames.append(plot_img)
            plt.close()
        # Ajouter une frame finale prolongée pour l'évolution IoU
        plt.figure(figsize=(4, 3))
        plt.plot(local_iou_evolution, marker="o", linestyle="-")
        plt.title(f"Obj {obj_idx+1} - Final")
        plt.xlabel("Étape")
        plt.ylabel("IoU")
        plt.ylim(0, 1)
        plt.grid(True)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        final_plot = Image.open(buf).convert("RGB")
        for _ in range(10):
            iou_evolution_frames.append(final_plot)
        plt.close()

    # Sauvegarde du GIF de détection
    # Ici, chaque frame est affichée pendant 200 ms, et le paramètre loop=1 permet de jouer le GIF une seule fois (donc ne boucle pas)
    detection_frames[0].save(
        output_gif_detection,
        format="GIF",
        append_images=detection_frames[1:],
        save_all=True,
        duration=200,
        loop=1
    )
    print(f"Le GIF de détection est sauvegardé dans {output_gif_detection}")

    # Sauvegarde du GIF montrant l'évolution de l'IoU
    iou_evolution_frames[0].save(
        output_gif_iou,
        format="GIF",
        append_images=iou_evolution_frames[1:],
        save_all=True,
        duration=200,
        loop=1
    )
    print(f"Le GIF de l'évolution de l'IoU est sauvegardé dans {output_gif_iou}")

if __name__ == "__main__":

    
    run_inference(
        image_path="DataSet/dog_test/imgs/dog_000321.jpg",
        annotation_path="DataSet/dog_test/annotations/dog_000321.xml",
        model_checkpoint_path="model_saves/DQN/ResNet50/model_epoch_36_DoubleDQN_Perso_2.pth",
        output_gif_detection="bbox_ev_3.gif",
        output_gif_iou="iou_ev_3.gif"
    )
