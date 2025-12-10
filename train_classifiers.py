import os
import numpy as np
import joblib
import argparse
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC

# --- Configuration ---
FEATURES_DIR = "features"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# Dictionnaire des classifieurs à entraîner
CLASSIFIERS = {
    "lda": LDA(),
    "knn": KNN(n_neighbors=5),
    "svm_linear": SVC(kernel='linear', probability=True),
    "svm_rbf": SVC(kernel='rbf', probability=True)
}

def train(feature_extractor_name):
    """
    Charge les données d'entraînement, applique la sélection de features,
    entraîne les classifieurs et sauvegarde les modèles.
    """
    # --- Chargement des données ---
    x_train_path = os.path.join(FEATURES_DIR, f"{feature_extractor_name}_X_train.npy")
    y_train_path = os.path.join(FEATURES_DIR, f"{feature_extractor_name}_y_train.npy")

    if not os.path.exists(x_train_path) or not os.path.exists(y_train_path):
        print(f"Erreur : Fichiers de features pour '{feature_extractor_name}' non trouvés.")
        print("Veuillez d'abord lancer 'extract_features.py'.")
        return

    print(f"Chargement des données pour '{feature_extractor_name}'...")
    X_train = np.load(x_train_path)
    y_train = np.load(y_train_path)

    # --- Entraînement des modèles ---
    for clf_name, clf in CLASSIFIERS.items():
        print(f"\nEntraînement du classifieur : {clf_name}...")
        
        pipeline = Pipeline([
            ('feature_selection', SelectKBest(score_func=chi2, k=50)),
            ('classification', clf)
        ])
        
        # Entraînement du pipeline
        pipeline.fit(X_train, y_train)
        
        # Sauvegarde du modèle entraîné
        model_filename = f"{feature_extractor_name}_{clf_name}.joblib"
        model_path = os.path.join(MODELS_DIR, model_filename)
        joblib.dump(pipeline, model_path)
        
        print(f"Modèle sauvegardé dans : {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraîner des classifieurs sur des features extraites.")
    parser.add_argument(
        "--features", 
        type=str, 
        required=True, 
        choices=['mobilenetv2', 'resnet50'],
        help="Nom de l'extracteur de features à utiliser (ex: mobilenetv2)."
    )
    args = parser.parse_args()
    
    train(args.features)
