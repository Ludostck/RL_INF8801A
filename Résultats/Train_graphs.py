import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def normalize_color(color):
    """
    Si la couleur est spécifiée sous forme d'une liste ou d'un tuple de 3 entiers (0-255),
    cette fonction normalise ces valeurs dans l'intervalle [0,1]. Sinon, renvoie la couleur telle quelle.
    """
    if isinstance(color, (list, tuple)) and len(color) == 3:
        # Vérifier si les valeurs sont de type entier et dans la plage 0-255
        if all(isinstance(c, int) and 0 <= c <= 255 for c in color):
            return tuple(c / 255 for c in color)
    return color

def plot_comparative_train_results(files_info, epoch_to_annotate=1):
    """
    files_info : liste de dictionnaires, chacun avec les clés suivantes :
        - "path" : chemin du fichier CSV.
        - "label": étiquette pour la courbe (ex: "Train AlexNet", "Train ResNet", etc.).
        - "color": (optionnel) couleur à utiliser pour la courbe.
                   Peut être une chaîne (ex: "red", "#00FF00") ou un tuple/list RGB (ex: (255, 0, 0)).
    epoch_to_annotate : l'epoch à partir duquel débuter l'affichage et l'annotation (par défaut 1, puisque tous commencent à 1).
    
    Les CSV de training doivent contenir la colonne "Epoch" ainsi que les métriques.
    Les colonnes commençant par "Std" (ex: "StdFinalIoU", "StdActionsPerDetection") sont ignorées.
    """
    # Charger et préparer chaque DataFrame
    for info in files_info:
        df = pd.read_csv(info['path'])
        if "Epoch" not in df.columns:
            raise ValueError(f"Le fichier {info['path']} doit contenir la colonne 'Epoch'")
        # Utiliser la colonne 'Epoch' comme axe des x
        df['x'] = df['Epoch']
        # Filtrer pour ne garder que les lignes à partir de epoch_to_annotate
        df = df[df['x'] >= epoch_to_annotate]
        info['df'] = df
        info['x'] = df['x']
    
    # Déterminer les colonnes communes en excluant "Epoch" et celles commençant par "Std"
    common_columns = set(files_info[0]['df'].columns)
    for info in files_info[1:]:
        common_columns = common_columns.intersection(set(info['df'].columns))
    
    excluded_cols = {"Epoch", "x"}  # "x" est une colonne temporaire interne
    common_columns = sorted([col for col in common_columns 
                             if col not in excluded_cols and not col.startswith("Std")])
    
    # Tracer pour chaque métrique commune
    for col in common_columns:
        plt.figure(figsize=(10, 6))
        for info in files_info:
            df = info['df']
            x = info['x']
            label = info['label']
            # Récupérer et normaliser la couleur si elle est spécifiée en RGB
            color = info.get("color", None)
            color = normalize_color(color) if color is not None else None

            plt.plot(x, df[col], marker='o', linestyle='-', label=label, color=color)
            
            # Annotation: recherche de epoch_to_annotate ou de la valeur la plus proche
            if epoch_to_annotate in x.values:
                y_val = df.loc[x == epoch_to_annotate, col].values[0]
                plt.text(epoch_to_annotate, y_val, f"{y_val:.2f}", fontsize=12, fontweight='bold',
                         bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
            else:
                diff = np.abs(x - epoch_to_annotate)
                closest_index = diff.idxmin()  # indice de la valeur la plus proche
                closest_epoch = x.loc[closest_index]
                y_val = df.loc[closest_index, col]
                plt.text(closest_epoch, y_val, f"{y_val:.2f}", fontsize=12, fontweight='bold',
                         bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
        
        plt.xlabel("Epoch")
        plt.ylabel(col)
        plt.title(f"Comparaison de {col}")
        plt.grid(True)
        plt.legend()
        plt.xlim(left=epoch_to_annotate)
        plt.show()




Base = "Résultats/CSV/"
# Exemple d'utilisation avec 3 fichiers et couleurs en format RGB
files_info = [
    {"path": Base+"training_metrics_QN_AlexNet.csv",        "label": "QN AlexNet",       "epoch_factor": 1,      "color": (25, 255, 25)},      
    {"path": Base+"training_metrics_QN_ResNet18.csv",         "label": "QN ResNet18",        "epoch_factor": 1,    "color": (0, 200, 0)},  
    {"path": Base+"training_metrics_QN_ResNet50.csv",           "label": "QN ResNet50", "epoch_factor": 1,      "color": (0, 70, 0)}  
]

plot_comparative_train_results(files_info, epoch_to_annotate=5)
