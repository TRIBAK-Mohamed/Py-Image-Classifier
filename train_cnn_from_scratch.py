import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.utils import image_dataset_from_directory
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os


# --- Configuration ---
INPUT_DIR = "features/augmented"
MODELS_DIR = "models"
RESULTS_DIR = "results"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20 # Nombre d'époques d'entraînement

def train_from_scratch():
    """
    Fonction principale pour entraîner et évaluer un CNN simple à partir de zéro.
    """
    if not os.path.exists(INPUT_DIR):
        print(f"Erreur : Le dossier '{INPUT_DIR}' n'existe pas. Veuillez d'abord lancer 'preprocess_images.py'.")
        return

    # --- 1. Chargement des données ---
    # On utilise une fonction Keras très pratique qui charge les images
    # et déduit les classes à partir de la structure des dossiers.
    print("Chargement des données d'images...")
    full_dataset = image_dataset_from_directory(
        INPUT_DIR,
        label_mode='int',
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=42,
        validation_split=0.2, # On sépare 20% pour le test
        subset='both' # On récupère à la fois train et validation (test)
    )
    train_dataset, test_dataset = full_dataset
    class_names = train_dataset.class_names
    print(f"Classes trouvées : {class_names}")

    # Optimisation des performances de chargement
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)

    # --- 2. Définition du modèle CNN ---
    print("Création du modèle CNN 'from scratch'...")
    model = Sequential([
        Input(shape=(224, 224, 3)),
        
        # Bloc Convolutif 1
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # Bloc Convolutif 2
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        # Bloc Convolutif 3
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # Aplatir les features pour les donner au classifieur
        Flatten(),
        
        # Couches de classification
        Dense(128, activation='relu'),
        Dropout(0.5), # Dropout pour limiter le sur-apprentissage
        Dense(len(class_names), activation='softmax') # Couche de sortie
    ])

    # --- 3. Compilation du modèle ---
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # --- 4. Entraînement du modèle ---
    print("\nDébut de l'entraînement...")
    history = model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=EPOCHS
    )

    # --- 5. Sauvegarde du modèle et des résultats ---
    print("Sauvegarde du modèle entraîné...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    model.save(os.path.join(MODELS_DIR, "cnn_from_scratch.keras"))

    # Sauvegarde du graphique de l'historique d'entraînement
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Accuracy (train)')
    plt.plot(history.history['val_accuracy'], label='Accuracy (test)')
    plt.title('Accuracy History')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Loss (train)')
    plt.plot(history.history['val_loss'], label='Loss (test)')
    plt.title('Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.savefig(os.path.join(RESULTS_DIR, "history_cnn_from_scratch.png"))
    print(f"Graphique de l'historique sauvegardé dans '{RESULTS_DIR}/history_cnn_from_scratch.png'")

    # --- 6. Évaluation finale ---
    print("\nÉvaluation finale sur l'ensemble de test...")
    y_true = np.concatenate([y for x, y in test_dataset], axis=0)
    y_pred_probs = model.predict(test_dataset)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("\nRapport de Classification :")
    print(classification_report(y_true, y_pred, target_names=class_names))

    print("Matrice de Confusion :")
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - CNN From Scratch')
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix_cnn_from_scratch.png"))
    print(f"Matrice de confusion sauvegardée dans '{RESULTS_DIR}/confusion_matrix_cnn_from_scratch.png'")
    plt.show()


if __name__ == "__main__":
    train_from_scratch()