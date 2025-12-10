import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import joblib
import cv2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess

# --- Configuration ---
IMG_SIZE = (224, 224)
MODELS_DIR = "models"
DATASET_DIR = "dataset"

# --- Helper Functions ---

def load_cnn_model():
    """Loads the custom CNN model."""
    model_path = os.path.join(MODELS_DIR, "cnn_from_scratch.keras")
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    return None

def load_ml_model(extractor_name, classifier_name):
    """Loads a specific Machine Learning model (pipeline)."""
    model_filename = f"{extractor_name}_{classifier_name}.joblib"
    model_path = os.path.join(MODELS_DIR, model_filename)
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

def get_class_names():
    """Attempts to retrieve class names from the dataset folder structure."""
    if os.path.exists(DATASET_DIR):
        return sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))])
    return ["Classe 0", "Classe 1", "Classe 2"] # Fallback

def extract_features_single_image(image, extractor_name):
    """
    Extracts features for a single image using a pre-trained CNN (MobileNetV2 or ResNet50).
    Reproduces the logic from extract_features.py but for one image.
    """
    img_array = np.array(image.resize(IMG_SIZE))
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension

    if extractor_name == 'mobilenetv2':
        from tensorflow.keras.applications import MobileNetV2
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg')
        img_preprocessed = mobilenet_preprocess(img_array)
    elif extractor_name == 'resnet50':
        from tensorflow.keras.applications import ResNet50
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg')
        img_preprocessed = resnet_preprocess(img_array)
    else:
        return None

    features = base_model.predict(img_preprocessed)
    return features.flatten() # Ensure 1D array for the classifier

# --- Streamlit App ---

st.set_page_config(page_title="Py-Image-Classifier", page_icon="📷", layout="wide")

st.title("📷 Py-Image-Classifier : Interface de Prédiction")
st.markdown("""
Cette application vous permet de tester vos modèles de classification d'images.
Choisissez un modèle, chargez une image (thermique ou autre) et visualisez le résultat.
""")

# --- Sidebar : Model Selection ---
st.sidebar.header("Configuration du Modèle")

model_type = st.sidebar.radio(
    "Type de Modèle :",
    ("CNN From Scratch", "Transfer Learning (Hybride)")
)

selected_model = None
class_names = get_class_names()

if model_type == "CNN From Scratch":
    st.sidebar.info("Modèle : Convolutional Neural Network (Custom)")
    with st.spinner("Chargement du modèle CNN..."):
        selected_model = load_cnn_model()
    
    if selected_model:
        st.sidebar.success("Modèle CNN chargé avec succès !")
    else:
        st.sidebar.error(f"Fichier 'cnn_from_scratch.keras' introuvable dans '{MODELS_DIR}'. Veuillez d'abord entraîner le modèle.")

elif model_type == "Transfer Learning (Hybride)":
    extractor_choice = st.sidebar.selectbox("Extracteur de Features :", ["mobilenetv2", "resnet50"])
    classifier_choice = st.sidebar.selectbox("Classifieur :", ["svm_linear", "svm_rbf", "knn", "lda"])
    
    st.sidebar.info(f"Pipeline : {extractor_choice} + {classifier_choice}")
    
    with st.spinner(f"Chargement du modèle {classifier_choice}..."):
        selected_model = load_ml_model(extractor_choice, classifier_choice)
    
    if selected_model:
        st.sidebar.success("Pipeline ML chargé avec succès !")
    else:
        st.sidebar.error(f"Modèle '{extractor_choice}_{classifier_choice}.joblib' introuvable. Avez-vous exécuté 'train_classifiers.py' ?")

# --- Main Area : Inference ---

st.header("Upload d'Image")
uploaded_file = st.file_uploader("Choisissez une image (JPG, PNG, JPEG)...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display Image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Image chargée', width=400)
    
    if st.button("Lancer la Prédiction"):
        if selected_model is None:
            st.error("Aucun modèle chargé. Veuillez vérifier la configuration dans la barre latérale.")
        else:
            with st.spinner("Analyse en cours..."):
                try:
                    if model_type == "CNN From Scratch":
                        # Preprocessing for CNN
                        img_array = image.resize(IMG_SIZE)
                        img_array = np.array(img_array)
                        img_extended = np.expand_dims(img_array, axis=0) # (1, 224, 224, 3)
                        
                        # Prediction
                        prediction_probs = selected_model.predict(img_extended)
                        predicted_class_idx = np.argmax(prediction_probs)
                        confidence = np.max(prediction_probs)
                        
                        predicted_label = class_names[predicted_class_idx] if predicted_class_idx < len(class_names) else f"Classe {predicted_class_idx}"
                        
                        st.success(f"**Classe Prédite : {predicted_label}**")
                        st.info(f"Confiance : {confidence:.2%}")
                        
                        # Show probabilities chart
                        st.bar_chart(dict(zip(class_names, prediction_probs[0])))

                    elif model_type == "Transfer Learning (Hybride)":
                        # 1. Feature Extraction
                        features = extract_features_single_image(image, extractor_choice)
                        
                        # 2. Classification
                        # Reshape features for user single sample (1, n_features)
                        features = features.reshape(1, -1)
                        
                        prediction_idx = selected_model.predict(features)[0]
                        predicted_label = class_names[prediction_idx] if prediction_idx < len(class_names) else f"Classe {prediction_idx}"
                        
                        st.success(f"**Classe Prédite : {predicted_label}**")
                        
                        # Try to get probabilities if supported
                        if hasattr(selected_model, "predict_proba"):
                            probs = selected_model.predict_proba(features)[0]
                            st.info(f"Confiance : {np.max(probs):.2%}")
                            st.bar_chart(dict(zip(class_names, probs)))
                        else:
                            st.warning("Ce classifieur ne fournit pas de probabilités de confiance.")

                except Exception as e:
                    st.error(f"Une erreur est survenue lors de la prédiction : {e}")

