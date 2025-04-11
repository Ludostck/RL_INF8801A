import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def normalize_color(color):
    """
    Si la couleur est spécifiée sous forme d'une liste ou d'un tuple de 3 entiers (0-255),
    cette fonction normalise ces valeurs dans l'intervalle [0,1]. Sinon, renvoie la couleur telle quelle.
    """
    if isinstance(color, (list, tuple)) and len(color) == 3:
        if all(isinstance(c, int) and 0 <= c <= 255 for c in color):
            return tuple(c / 255 for c in color)
    return color

def plot_comparative_test_results(files_info, epoch_to_annotate=11):
    """
    files_info : liste de dictionnaires, chacun avec les clés suivantes :
        - "path" : chemin du fichier CSV.
        - "label": étiquette pour la courbe (ex: "Base 1", "DQN", etc.).
        - "epoch_factor": facteur de conversion pour model_epoch (1 pour BASE, 0.5 pour DQN).
        - "color": (optionnel) couleur à utiliser pour la courbe.
                   Peut être une chaîne (ex: "red", "#00FF00") ou un tuple/list RGB (ex: (255, 0, 0)).
        - "count_csv": (obligatoire) 0 ou 1.
             Si 1, le label est modifié pour inclure le nombre de lignes filtrées.
             Si 0, le CSV est quand même traité (pour l'adaptation des axes), 
             mais la courbe sera tracée en blanc.
    epoch_to_annotate : l'epoch (après conversion) à partir duquel débuter l'affichage et l'annotation.
    
    Seules les colonnes suivantes seront affichées si elles existent : 
        - average_final_iou 
        - actions_per_detection 
        - percent_actions_improved
    """
    # Chargement et filtrage de chaque CSV
    for info in files_info:
        df = pd.read_csv(info['path'])
        if "model_epoch" not in df.columns:
            raise ValueError(f"Le fichier {info['path']} doit contenir la colonne 'model_epoch'")
        # Calcul de la nouvelle colonne x selon l'epoch_factor
        df['x'] = df['model_epoch'] * info['epoch_factor']
        # Filtrer pour conserver x >= epoch_to_annotate
        df = df[df['x'] >= epoch_to_annotate]
        info['df'] = df
        info['x'] = df['x']
        
        # Modification du label en fonction de count_csv
        if info.get("count_csv", 0) == 1:
            info['plot_label'] = f"{info['label']} (n={len(df)})"
        else:
            info['plot_label'] = info['label']
    
    # Calcul des colonnes communes à tous les CSV en excluant les colonnes non pertinentes
    common_columns = set(files_info[0]['df'].columns)
    for info in files_info[1:]:
        common_columns = common_columns.intersection(set(info['df'].columns))
    excluded_cols = {"model_epoch", "model_file", "x"}
    common_columns = sorted([col for col in common_columns if col not in excluded_cols])
    
    # On veut afficher uniquement ces 3 colonnes s'il elles existent dans les fichiers
    desired_columns = ["average_final_iou", "avg_actions_per_detection", "percent_actions_improved"]
    columns_to_plot = [col for col in desired_columns if col in common_columns]
    
    if not columns_to_plot:
        print("Aucune des colonnes désirées n'a été trouvée dans les CSV.")
        return

    # On réordonne la liste pour que les CSV avec "count_csv" à 0 soient tracés en premier
    files_info_sorted = sorted(files_info, key=lambda x: x.get("count_csv", 0))
    
    # Tracer pour chaque colonne désirée
    for col in columns_to_plot:
        plt.figure(figsize=(10, 6))
        for info in files_info_sorted:
            df = info['df']
            x = info['x']
            label = info['plot_label']
            # Pour le CSV avec count_csv=0, tracer en blanc (invisible sur fond blanc)
            if info.get("count_csv", 0) == 0:
                plot_color = "white"
            else:
                color = info.get("color", None)
                plot_color = normalize_color(color) if color is not None else None
            
            plt.plot(x, df[col], marker='o', linestyle='-', label=label, color=plot_color)
        
        plt.xlabel("Epoch")
        plt.ylabel(col)
        plt.title(f"Comparaison de {col}")
        plt.grid(True)
        plt.legend(loc='upper right')
        plt.xlim(left=epoch_to_annotate)
        plt.tight_layout()
        plt.show()

# Exemple d'utilisation avec chaque CSV ayant son propre flag
Base = "Résultats/CSV/"
files_info = [
    {"path": Base+"TEST_QN_AlexNet.csv",  "label": "QN AlexNet",    "epoch_factor": 1, "color": (130, 255, 130), "count_csv": 1},
    {"path": Base+"TEST_QN_ResNet18.csv",   "label": "QN ResNet18",   "epoch_factor": 1, "color": (15, 255, 15),   "count_csv": 1},
    {"path": Base+"TEST_QN_ResNet50.csv",    "label": "QN ResNet50",   "epoch_factor": 1, "color": (0, 100, 0),   "count_csv": 1},
    {"path": Base+"TEST_DQN_AlexNet.csv",    "label": "DQN AlexNet",   "epoch_factor": 0.5, "color": (230, 200, 255), "count_csv": 1},
    {"path": Base+"TEST_DQN_ResNet18.csv",   "label": "DQN ResNet18",  "epoch_factor": 0.5, "color": (180, 100, 255),  "count_csv": 1},
    {"path": Base+"TEST_DQN_ResNet50.csv",   "label": "DQN ResNet50",  "epoch_factor": 0.5, "color": (90, 0, 160),    "count_csv": 1}
]

plot_comparative_test_results(files_info, epoch_to_annotate=11)
