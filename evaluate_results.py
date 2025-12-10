import os
import numpy as np
import joblib
import json
import argparse
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
FEATURES_DIR = "features"
MODELS_DIR = "models"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def save_confusion_matrix_plot(conf_matrix, model_name, output_dir, class_names):
    """
    Crée une heatmap de la matrice de confusion et la sauvegarde en tant qu'image.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_matrix, 
        annot=True,     # Affiche les nombres dans les cases
        fmt='d',        # Format des nombres (entiers)
        cmap='Blues',   # Palette de couleurs
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title(f'Matrice de Confusion - {model_name}', fontsize=16)
    plt.ylabel('Vraie Classe (True Label)', fontsize=12)
    plt.xlabel('Classe Prédite (Predicted Label)', fontsize=12)
    
    plot_filename = f"confusion_matrix_{model_name}.png"
    plot_path = os.path.join(output_dir, plot_filename)
    
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close() # Fermer la figure pour libérer la mémoire
    print(f"  -> Matrice de confusion sauvegardée dans : {plot_path}")


def evaluate(feature_extractor_name):
    """
    Charge les données de test et les modèles, calcule les métriques,
    visualise la matrice de confusion et génère un rapport JSON.
    """
    # --- Chargement des données de test ---
    x_test_path = os.path.join(FEATURES_DIR, f"{feature_extractor_name}_X_test.npy")
    y_test_path = os.path.join(FEATURES_DIR, f"{feature_extractor_name}_y_test.npy")

    if not os.path.exists(x_test_path) or not os.path.exists(y_test_path):
        print(f"Erreur : Fichiers de test pour '{feature_extractor_name}' non trouvés.")
        return

    print(f"Chargement des données de test pour '{feature_extractor_name}'...")
    X_test = np.load(x_test_path)
    y_test = np.load(y_test_path)

   
    augmented_dir = os.path.join(FEATURES_DIR, "augmented")
    class_names = sorted([d for d in os.listdir(augmented_dir) if os.path.isdir(os.path.join(augmented_dir, d))])
    print(f"Classes détectées : {class_names}")

    # Évaluation des modèles 
    report = {}
    model_files = [f for f in os.listdir(MODELS_DIR) if f.startswith(feature_extractor_name) and f.endswith(".joblib")]

    if not model_files:
        print(f"Aucun modèle trouvé pour '{feature_extractor_name}'. Veuillez lancer 'train_classifiers.py'.")
        return

    for model_file in model_files:
        model_path = os.path.join(MODELS_DIR, model_file)
        model_name = os.path.splitext(model_file)[0]
        
        print(f"\nÉvaluation du modèle : {model_name}...")
        
        model = joblib.load(model_path)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        
        print(f"  Accuracy: {accuracy:.4f}")


        save_confusion_matrix_plot(conf_matrix, model_name, RESULTS_DIR, class_names)
        
        report[model_name] = {
            "accuracy": accuracy,
            "confusion_matrix": conf_matrix.tolist(), 
            "classification_report": class_report
        }

    report_filename = f"report_{feature_extractor_name}.json"
    report_path = os.path.join(RESULTS_DIR, report_filename)
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
        
    print(f"\nRapport d'évaluation complet sauvegardé dans : {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Évaluer les modèles entraînés.")
    parser.add_argument(
        "--features", 
        type=str, 
        required=True, 
        choices=['mobilenetv2', 'resnet50'],
        help="Nom de l'extracteur de features à évaluer (ex: mobilenetv2)."
    )
    args = parser.parse_args()
    
    evaluate(args.features)