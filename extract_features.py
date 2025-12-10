import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2, ResNet50, imagenet_utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# --- Configuration ---
INPUT_DIR = "features/augmented"
TARGET_SIZE = (224, 224)


MODELS = {
    "mobilenetv2": MobileNetV2(weights='imagenet', include_top=False, pooling='avg'),
    "resnet50": ResNet50(weights='imagenet', include_top=False, pooling='avg') 
}

def extract_features():
    """
    Charge les images, extrait les features avec chaque modèle CNN,
    et sauvegarde les jeux de données train/test.
    """
    if not os.path.exists(INPUT_DIR):
        print(f"Erreur : Le dossier '{INPUT_DIR}' n'existe pas. Veuillez d'abord lancer 'preprocess_images.py'.")
        return

    image_paths = []
    labels = []
    class_names = [d for d in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, d))]
    
    if not class_names:
        print(f"Erreur : Aucun sous-dossier (classe) trouvé dans {INPUT_DIR}.")
        return

    for class_name in class_names:
        class_dir = os.path.join(INPUT_DIR, class_name)
        for image_name in os.listdir(class_dir):
            image_paths.append(os.path.join(class_dir, image_name))
            labels.append(class_name)

    le = LabelEncoder()
    encoded_labels = le.fit_transform(labels)

    for model_name, model in MODELS.items():
        print("-" * 50)
        print(f"Extraction des features avec le modèle : {model_name}...")
        
        all_features = []
        for image_path in tqdm(image_paths, desc=f"Extraction {model_name}"):
            img = load_img(image_path, target_size=TARGET_SIZE)
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            
            preprocessed_img = imagenet_utils.preprocess_input(img_array, mode='caffe')

            features = model.predict(preprocessed_img, verbose=0)
            all_features.append(features.flatten())

        X = np.array(all_features)
        y = np.array(encoded_labels)

        # Train/Test 
        print("Découpage des données en ensembles d'entraînement et de test (80%/20%)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )

        # Sauvegarde des fichiers .npy 
        output_dir = "features"
        np.save(os.path.join(output_dir, f"{model_name}_X_train.npy"), X_train)
        np.save(os.path.join(output_dir, f"{model_name}_y_train.npy"), y_train)
        np.save(os.path.join(output_dir, f"{model_name}_X_test.npy"), X_test)
        np.save(os.path.join(output_dir, f"{model_name}_y_test.npy"), y_test)
        
        print(f"Features pour {model_name} sauvegardées avec succès.")
        print("-" * 50)

if __name__ == "__main__":
    extract_features()