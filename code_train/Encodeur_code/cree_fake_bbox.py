import os
import random
import xml.etree.ElementTree as ET
from PIL import Image
import pandas as pd
from tqdm import tqdm
# Définition des actions possibles
ACTIONS = ["left", "right", "up", "down", "zoom_in", "widen", "elongate"]
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
    "West_Highland_white_terrier", "Whippet", "Wire-haired_fox_terrier", "Yorkshire_terrier", 'dog',' dog',' dog ','dog '
]
def apply_action(bbox, action, img_width, img_height, min_size=10, alpha=0.2):
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1

    dx = int(alpha * w)
    dy = int(alpha * h)

    if action == "left":
        x1 = max(0, x1 - dx)
        x2 = max(x1 + min_size, x2 - dx)
    elif action == "right":
        x1 = min(img_width - min_size, x1 + dx)
        x2 = min(img_width, x2 + dx)
    elif action == "up":
        y1 = max(0, y1 - dy)
        y2 = max(y1 + min_size, y2 - dy)
    elif action == "down":
        y1 = min(img_height - min_size, y1 + dy)
        y2 = min(img_height, y2 + dy)
    elif action == "zoom_in":
        x1 = x1 + dx
        x2 = x2 - dx
        y1 = y1 + dy
        y2 = y2 - dy
    elif action == "zoom_out":
        x1 = max(0, x1 - dx)
        x2 = min(img_width, x2 + dx)
        y1 = max(0, y1 - dy)
        y2 = min(img_height, y2 + dy)
    elif action == "widen":
        x1 = max(0, x1 - dx)
        x2 = min(img_width, x2 + dx)
    elif action == "elongate":
        y1 = max(0, y1 - dy)
        y2 = min(img_height, y2 + dy)

    # Clamp to valid box
    x1, x2 = sorted([max(0, min(img_width, x1)), max(0, min(img_width, x2))])
    y1, y2 = sorted([max(0, min(img_height, y1)), max(0, min(img_height, y2))])

    if x2 - x1 < min_size or y2 - y1 < min_size:
        return bbox  # Return original if too small

    return [x1, y1, x2, y2]

def compute_iou(boxA, boxB):
    # Intersection
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter_area = max(0, xB - xA) * max(0, yB - yA)

    # Union
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union_area = boxA_area + boxB_area - inter_area

    if union_area == 0:
        return 0.0
    return inter_area / union_area

# Répertoires
img_dir = "DataSet/dog_unique_train/img"
ann_dir = "DataSet/dog_unique_train/annotations"

# Stockage des résultats
results = []

for ann_file in tqdm(os.listdir(ann_dir)):
    if not ann_file.endswith(".xml"):
        continue

    # Chemins
    xml_path = os.path.join(ann_dir, ann_file)
    img_filename = os.path.splitext(ann_file)[0] + ".jpg"
    img_path = os.path.join(img_dir, img_filename)

    if not os.path.exists(img_path):
        continue

    # Image size
    with Image.open(img_path) as img:
        W, H = img.size

    # Lire la vraie bbox (il y en a exactement une)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    gt_bbox = None
    # Itérer sur tous les objets trouvés
    for obj in root.findall("object"):
        # Récupérer le nom de l'objet
        obj_name = obj.find("name").text
        # Vérifier si le nom de l'objet est dans la liste dog_classes
        
        if obj_name in dog_classes:
            # Extraire la bounding box
            bndbox = obj.find("bndbox")
            x1 = int(round(float(bndbox.find("xmin").text)))
            y1 = int(round(float(bndbox.find("ymin").text)))
            x2 = int(round(float(bndbox.find("xmax").text)))
            y2 = int(round(float(bndbox.find("ymax").text)))
            gt_bbox = [x1, y1, x2, y2]
            # Traiter la bounding box (par exemple, l'afficher ou la stocker)
    if gt_bbox is None : 
        print(f"aucun objet n'appartient pas aux classes recherchées.")
        for obj in root.findall("object"):

            print(obj.find("name").text)
        continue

    # 1. BBOX légèrement modifiée (jitter)
    jitter = 0.06
    dx = int(jitter * (x2 - x1))
    dy = int(jitter * (y2 - y1))
    jitter_bbox = [
        max(0, x1 - random.randint(0, dx)),
        max(0, y1 - random.randint(0, dy)),
        min(W, x2 + random.randint(0, dx)),
        min(H, y2 + random.randint(0, dy)),
    ]
    iou_jitter = compute_iou(gt_bbox, jitter_bbox)
    results.append([img_filename, "jitter1", jitter_bbox, iou_jitter])

    jitter = 0.06
    dx = int(jitter * (x2 - x1))
    dy = int(jitter * (y2 - y1))
    jitter_bbox = [
        max(0, x1 - random.randint(0, dx)),
        max(0, y1 - random.randint(0, dy)),
        min(W, x2 + random.randint(0, dx)),
        min(H, y2 + random.randint(0, dy)),
    ]
    iou_jitter = compute_iou(gt_bbox, jitter_bbox)
    results.append([img_filename, "jitter2", jitter_bbox, iou_jitter])

 

    # 2. BBOX transformée via actions successives
    action_bbox = [0, 0, W, H]
    for _ in range(4):
        action = random.choice(ACTIONS)
        action_bbox = apply_action(action_bbox, action, W, H)
    iou_action = compute_iou(gt_bbox, action_bbox)
    results.append([img_filename, "action_seq1", action_bbox, iou_action])

    # 2. BBOX transformée via actions successives
    action_bbox = [0, 0, W, H]
    for _ in range(4):
        action = random.choice(ACTIONS)
        action_bbox = apply_action(action_bbox, action, W, H)
    iou_action = compute_iou(gt_bbox, action_bbox)
    results.append([img_filename, "action_seq2", action_bbox, iou_action])
    # 2. BBOX transformée via actions successives
    action_bbox = [0, 0, W, H]
    for _ in range(4):
        action = random.choice(ACTIONS)
        action_bbox = apply_action(action_bbox, action, W, H)
    iou_action = compute_iou(gt_bbox, action_bbox)
    results.append([img_filename, "action_seq3", action_bbox, iou_action])

    # 3. BBOX complètement aléatoire (10% surface)
    val = random.uniform(0.075, 0.2)

    target_area = int(val * W * H)
    for _ in range(30):  # Try multiple times until area is close
        rw = random.randint(int(W * 0.1), int(W * 0.5))
        rh = target_area // rw
        if rh > H:
            continue
        x1 = random.randint(0, W - rw)
        y1 = random.randint(0, H - rh)
        x2 = x1 + rw
        y2 = y1 + rh
        rand_bbox = [x1, y1, x2, y2]
        break
    iou_rand = compute_iou(gt_bbox, rand_bbox)
    results.append([img_filename, "random1", rand_bbox, iou_rand])

    # 3. BBOX complètement aléatoire (10% surface)
    val = random.uniform(0.2, 0.3)

    target_area = int(val * W * H)
    for _ in range(30):  # Try multiple times until area is close
        rw = random.randint(int(W * 0.1), int(W * 0.5))
        rh = target_area // rw
        if rh > H:
            continue
        x1 = random.randint(0, W - rw)
        y1 = random.randint(0, H - rh)
        x2 = x1 + rw
        y2 = y1 + rh
        rand_bbox = [x1, y1, x2, y2]
        break
    iou_rand = compute_iou(gt_bbox, rand_bbox)
    results.append([img_filename, "random2", rand_bbox, iou_rand])

    # 3. BBOX complètement aléatoire (10% surface)
    val = random.uniform(0.3, 0.4)

    target_area = int(val * W * H)
    for _ in range(30):  # Try multiple times until area is close
        rw = random.randint(int(W * 0.1), int(W * 0.5))
        rh = target_area // rw
        if rh > H:
            continue
        x1 = random.randint(0, W - rw)
        y1 = random.randint(0, H - rh)
        x2 = x1 + rw
        y2 = y1 + rh
        rand_bbox = [x1, y1, x2, y2]
        break
    iou_rand = compute_iou(gt_bbox, rand_bbox)
    results.append([img_filename, "random3", rand_bbox, iou_rand])

    # 4. BBOX à partir du groundtruth avec 1 ou 2 actions
    modified_bbox = gt_bbox.copy()
    n_actions = random.choice([1, 2])
    for _ in range(n_actions):
        action = random.choice(ACTIONS)
        modified_bbox = apply_action(modified_bbox, action, W, H)
    iou_modified = compute_iou(gt_bbox, modified_bbox)
    results.append([img_filename, f"gt_action_{n_actions}step1", modified_bbox, iou_modified])

    # 4. BBOX à partir du groundtruth avec 1 ou 2 actions
    modified_bbox = gt_bbox.copy()
    n_actions = random.choice([1, 2])
    for _ in range(n_actions):
        action = random.choice(ACTIONS)
        modified_bbox = apply_action(modified_bbox, action, W, H)
    iou_modified = compute_iou(gt_bbox, modified_bbox)
    results.append([img_filename, f"gt_action_{n_actions}step2", modified_bbox, iou_modified])

    # 4. BBOX à partir du groundtruth avec 1 ou 2 actions
    modified_bbox = gt_bbox.copy()
    n_actions = random.choice([1, 2])
    for _ in range(n_actions):
        action = random.choice(ACTIONS)
        modified_bbox = apply_action(modified_bbox, action, W, H)
    iou_modified = compute_iou(gt_bbox, modified_bbox)
    results.append([img_filename, f"gt_action_{n_actions}step3", modified_bbox, iou_modified])

# Conversion en DataFrame pour analyse ou sauvegarde
df = pd.DataFrame(results, columns=["image", "type", "bbox", "iou"])
# Affichage simple
print(df.head())

# Optionnel : enregistrer les résultats dans un fichier CSV
df.to_csv("code_train/Encodeur_code/bbox_iou_dataset_train.csv", index=False)

 