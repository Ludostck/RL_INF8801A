import random
from torchvision import transforms
import torch
from PIL import ImageDraw
import xml.etree.ElementTree as ET


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Actions possibles
ACTIONS = ["left", "right", "up", "down", "zoom_in", "zoom_out", "widen", "elongate", "trigger"]

def tensor_to_pil(tensor):
    """
    Convertie : tensor -> PIL Image.
    """
    to_pil = transforms.ToPILImage()
    return to_pil(tensor.squeeze(0).cpu())


def pil_to_tensor(pil_image):
    """
    Convertie : PIL Image -> tensor.
    """
    to_tensor = transforms.ToTensor()
    return to_tensor(pil_image)



def apply_action(bbox, action, img_width, img_height, min_size=10, alpha=0.2):
    """
    Applique une action sur une boîte en ajustant ses coordonnées.

    Args:
        bbox : Liste contenant les coordonnées [x1, y1, x2, y2] de la boîte englobante.
        action : Action à appliquer (une des ACTIONS).
        img_width : Largeur de l'image.
        img_height : Hauteur de l'image.
        min_size : Taille minimale pour la largeur et la hauteur de la boîte englobante.
        alpha : Facteur d'échelle pour l'action.

    Returns:
        Liste contenant les nouvelles coordonnées [x1, y1, x2, y2] de la boîte englobante.
    """
    x1, y1, x2, y2 = bbox[:]
    width, height = x2 - x1, y2 - y1

    # Appliquer les transformations en fonction de l'action
    if action == "left":
        x1 -= alpha * width
        x2 -= alpha * width
    elif action == "right":
        x1 += alpha * width
        x2 += alpha * width
    elif action == "up":
        y1 -= alpha * height
        y2 -= alpha * height
    elif action == "down":
        y1 += alpha * height
        y2 += alpha * height
    elif action == "zoom_in":
        x1 += alpha * width
        x2 -= alpha * width
        y1 += alpha * height
        y2 -= alpha * height
    elif action == "zoom_out":
        x1 -= alpha * width
        x2 += alpha * width
        y1 -= alpha * height
        y2 += alpha * height
    elif action == "widen":
        x1 -= alpha * width
        x2 += alpha * width
    elif action == "elongate":
        y1 -= alpha * height
        y2 += alpha * height

    # Contraindre les coordonnées aux limites de l'image
    if img_width is not None and img_height is not None:
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_width, x2)
        y2 = min(img_height, y2)

    # Assurer une taille minimale de la boîte englobante
    if x2 - x1 < min_size:
        center_x = (x1 + x2) / 2
        x1 = center_x - min_size / 2
        x2 = center_x + min_size / 2

    if y2 - y1 < min_size:
        center_y = (y1 + y2) / 2
        y1 = center_y - min_size / 2
        y2 = center_y + min_size / 2

    # Réajuster si la boîte dépasse les limites après avoir imposé la taille minimale
    if img_width is not None and img_height is not None:
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_width, x2)
        y2 = min(img_height, y2)

    return [x1, y1, x2, y2]


def calculate_iou(bbox, ground_truth):
    """
    Calcule l'Intersection over Union (IoU) entre une boîte englobante et la vérité terrain.

    Args:
        bbox : Liste contenant les coordonnées [x1, y1, x2, y2] de la boîte englobante prédite.
        ground_truth : Dictionnaire contenant 'xmin', 'ymin', 'xmax', 'ymax' de la boîte englobante de la vérité terrain.

    Returns:
        Valeur de l'IoU sous forme de flottant.
    """
    x1, y1, x2, y2 = bbox[:]
    gx1, gy1, gx2, gy2 = ground_truth["xmin"], ground_truth["ymin"], ground_truth["xmax"], ground_truth["ymax"]

    # Calculer l'intersection
    intersection_width = max(0, min(x2, gx2) - max(x1, gx1))
    intersection_height = max(0, min(y2, gy2) - max(y1, gy1))
    intersection_area = intersection_width * intersection_height

    # Calculer l'union
    bbox_area = (x2 - x1) * (y2 - y1)
    ground_truth_area = (gx2 - gx1) * (gy2 - gy1)
    union_area = bbox_area + ground_truth_area - intersection_area

    # Retourner l'IoU
    return intersection_area / union_area if union_area > 0 else 0


def get_history_tensor(history_deque):
    """
    Convertie une deque d'actions historiques en un tenseur.

    Args:
        history_deque : Deque contenant les vecteurs d'actions encodés en one-hot.

    Returns:
        Tenseur représentant l'historique empilé.
    """
    # Empiler tous les vecteurs d'action et les aplatir
    history_tensor = torch.cat(list(history_deque)).unsqueeze(0)  # Forme : (1, 90)
    return history_tensor.to(device)


def calculate_reward(current_iou, prev_iou, action_idx, eta=3, tau=0.6):
    """
    Calcule la récompense
    """
    if action_idx == 8:
        # Action de déclenchement : utiliser une fonction de récompense progressive.
        if current_iou < tau :
            reward = -3
        else:
            reward = 3 
        return reward
    else:
        # Pour les actions non déclenchantes, la récompense est proportionnelle au changement d'IoU.
        iou_diff = current_iou - prev_iou
        if iou_diff > 0:
            reward = 1
        else : 
            reward = -1
        return reward


def get_best_action(bbox, ground_truth, img_width, img_height):
    """
    Détermine la meilleure action à effectuer en se basant sur la boîte englobante actuelle et la vérité terrain.

    Args:
        bbox : Boîte englobante actuelle sous forme de [x1, y1, x2, y2].
        ground_truth : Boîte englobante de la vérité terrain.
        img_width : Largeur de l'image.
        img_height : Hauteur de l'image.

    Returns:
        Indice de la meilleure action sélectionnée.
    """
    initial_iou = calculate_iou(bbox, ground_truth)
    best_actions = []
    trigger_threshold = 0.6

    for idx, action in enumerate(ACTIONS):
        if action == "trigger":
            # Vérifier si l'action 'trigger' est valide
            if initial_iou > trigger_threshold:
                return 8
        else:
            # Calculer l'IoU pour les autres actions
            new_iou = calculate_iou(apply_action(bbox, action, img_width, img_height), ground_truth)
            if new_iou > initial_iou:
                best_actions.append((idx, new_iou))

    if best_actions:
        indices = [action[0] for action in best_actions]
        weights = [action[1] for action in best_actions]

        # Effectuer un choix aléatoire pondéré
        selected_idx = random.choices(indices, weights=weights, k=1)[0]
        return selected_idx
    else:
        indices = list(range(8))
        return random.choices(indices)[0]




def calculate_area(bbox):
    """
    Calcule l'aire d'une boîte englobante.

    Args:
        bbox : Dictionnaire contenant 'xmin', 'ymin', 'xmax', 'ymax'.

    Returns:
        Aire de la boîte englobante sous forme de flottant.
    """
    width = bbox['xmax'] - bbox['xmin']
    height = bbox['ymax'] - bbox['ymin']
    return width * height


def sort_bounding_boxes_by_area(bounding_boxes):
    """
    Trie les boîtes englobantes par leur aire en ordre croissant.

    Args:
        bounding_boxes : Liste de boîtes englobantes, chacune étant un dictionnaire.

    Returns:
        Liste de boîtes englobantes triées par aire.
    """
    sorted_boxes = sorted(bounding_boxes, key=calculate_area)
    return sorted_boxes


def draw_cross(image_pil, bbox, color=(0, 0, 0), w=2, h=2):
    """
    Dessine une croix sur l'image au centre de la boîte englobante.

    Args:
        image_pil : Objet image PIL.
        bbox : Liste contenant les coordonnées [x1, y1, x2, y2].
        color : Tuple représentant la couleur de la croix.
        w : Largeur de la ligne verticale.
        h : Hauteur de la ligne horizontale.
    """
    draw = ImageDraw.Draw(image_pil)
    x1, y1, x2, y2 = bbox
    # Dessiner des lignes d'un coin à l'autre (lignes de la croix)
    draw.line([((x1 + x2)/2, y1), ((x1 + x2)/2, y2)], fill=color, width=w)
    draw.line([(x1, (y1 + y2)/2), (x2, (y1 + y2)/2)], fill=color, width=h)


def add_border_to_bbox(bbox, border_size, img_width, img_height):
    """
    Ajoute une bordure à la boîte englobante, en s'assurant qu'elle reste à l'intérieur des limites de l'image.

    Args:
        bbox : Liste contenant les coordonnées [x1, y1, x2, y2].
        border_size : Taille de la bordure à ajouter.
        img_width : Largeur de l'image.
        img_height : Hauteur de l'image.

    Returns:
        Dictionnaire contenant 'xmin', 'ymin', 'xmax', 'ymax' avec les bordures ajoutées.
    """
    return {
        "xmin": max(0, bbox[0] - border_size),
        "ymin": max(0, bbox[1] - border_size),
        "xmax": min(img_width, bbox[2] + border_size),
        "ymax": min(img_height, bbox[3] + border_size),
    }


def get_features(bbox, cnn, image_tensor, device, transform):
    """
    Extrait les caractéristiques du tenseur d'image en utilisant un CNN pour la boîte englobante donnée.

    Args:
        bbox : Liste contenant les coordonnées [x1, y1, x2, y2] de la boîte englobante.
        cnn : Modèle CNN utilisé pour l'extraction de caractéristiques.
        image_tensor : Tenseur représentant l'image.
        device : Appareil Torch (CPU ou GPU).
        transform : Transformation à appliquer à l'image rognée.

    Returns:
        Tenseur de caractéristiques extrait par le CNN.
    """
    img_height, img_width = image_tensor.shape[1:3]
    bbox_with_border = add_border_to_bbox(bbox, 16, img_width, img_height)

    # Rogner la région de l'image
    cropped_region = image_tensor[:, 
                                  int(bbox_with_border["ymin"]):int(bbox_with_border["ymax"]), 
                                  int(bbox_with_border["xmin"]):int(bbox_with_border["xmax"])]

    cropped_region_pil = tensor_to_pil(cropped_region)
    cropped_region_transformed = transform(cropped_region_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        features = cnn(cropped_region_transformed)
    
    return features


def load_annotations(xml_file):
    """
    Charge les bounding boxes de chiens à partir d’un fichier XML, quel que soit le format (VOC ou ImageNet).

    Args:
        xml_file : chemin vers le fichier XML
        dog_classes : liste des noms de classes de chiens acceptés (ex: ["dog", "Chihuahua", "Beagle"]). 
                     Si None, prend "dog" comme seul nom.

    Returns:
        Liste de dictionnaires contenant les BBox des chiens : {"name", "xmin", "ymin", "xmax", "ymax"}
    """

    dog_classes = [
    "Affenpinscher", "Afghan_hound", "African_hunting_dog", "Airedale", "American_Staffordshire_terrier",
    "Appenzeller", "Australian_terrier", "Basenji", "Basset", "Beagle", "Bedlington_terrier",
    "Bernese_mountain_dog", "Black-and-tan_coonhound", "Blenheim_spaniel", "Bloodhound", "Bluetick",
    "Border_collie", "Border_terrier", "Borzoi", "Boston_bull", "Bouvier_des_Flandres", "Boxer",
    "Brabancon_griffon", "Briard", "Brittany_spaniel", "Bull_mastiff", "Cairn", "Cardigan", "Chesapeake_Bay_retriever",
    "Chihuahua", "Chow", "Clumber", "Cocker_spaniel", "Collie", "Curly-coated_retriever", "Dandie_Dinmont",
    "Dhole", "Dingo", "Doberman", "English_foxhound", "English_setter", "English_springer", "EntleBucher",
    "Eskimo_dog", "Flat-coated_retriever", "French_bulldog", "German_shepherd", "German_short-haired_pointer",
    "Giant_schnauzer", "Golden_retriever", "Gordon_setter", "Great_Dane", "Great_Pyrenees", "Greater_Swiss_Mountain_dog",
    "Groenendael", "Ibizan_hound", "Irish_setter", "Irish_terrier", "Irish_water_spaniel", "Irish_wolfhound",
    "Italian_greyhound", "Japanese_spaniel", "Keeshond", "Kelpie", "Kerry_blue_terrier", "Komondor",
    "Kuvasz", "Labrador_retriever", "Lakeland_terrier", "Leonberg", "Lhasa", "Malamute", "Malinois",
    "Maltese_dog", "Mexican_hairless", "Miniature_pinscher", "Miniature_poodle", "Miniature_schnauzer",
    "Newfoundland", "Norfolk_terrier", "Norwegian_elkhound", "Norwich_terrier", "Old_English_sheepdog",
    "Otterhound", "Papillon", "Pekinese", "Pembroke", "Petit_basset_griffon_Vendeen", "Pharaoh_hound",
    "Plott", "Pointer", "Pomeranian", "Pug", "Redbone", "Rhodesian_ridgeback", "Rottweiler",
    "Saint_Bernard", "Saluki", "Samoyed", "Schipperke", "Scotch_terrier", "Scottish_deerhound",
    "Sealyham_terrier", "Shetland_sheepdog", "Shih-Tzu", "Siberian_husky", "Silky_terrier", "Soft-coated_wheaten_terrier",
    "Staffordshire_bullterrier", "Standard_poodle", "Standard_schnauzer", "Sussex_spaniel", "Tibetan_mastiff",
    "Tibetan_terrier", "Toy_poodle", "Toy_terrier", "Vizsla", "Walker_hound", "Weimaraner", "Welsh_springer_spaniel",
    "West_Highland_white_terrier", "Whippet", "Wire-haired_fox_terrier", "Yorkshire_terrier","dog"
    ]

    if dog_classes is None:
        dog_classes = ["dog"]  # par défaut

    dog_classes = [cls.lower() for cls in dog_classes]  # pour comparer en minuscule

    tree = ET.parse(xml_file)
    root = tree.getroot()
    objects = root.findall("object")
    boxes = []

    for obj in objects:
        cls_name = obj.find("name").text.strip()
        if cls_name.lower() not in dog_classes:
            continue  # ignorer si ce n'est pas un chien

        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue

        xmin = int(round(float(bndbox.find("xmin").text)))
        ymin = int(round(float(bndbox.find("ymin").text)))
        xmax = int(round(float(bndbox.find("xmax").text)))
        ymax = int(round(float(bndbox.find("ymax").text)))

        boxes.append({
            "name": cls_name,
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax
        })

    return boxes
