# Py-Image-Classifier

**Py-Image-Classifier** is a  Computer Vision project applied to predictive maintenance in an industrial setting.

It is designed to **automatically detect and classify defects in electric motors** (overheating, wear, etc.) using images (specifically thermal images), enabling failure anticipation.

This modular pipeline allows for training high-performance models using two distinct approaches:
1.  **Transfer Learning**: Feature extraction via pre-trained deep neural networks (such as MobileNetV2 and ResNet50) coupled with classical classifiers.
2.  **CNN from Scratch**: End-to-end training of a custom convolutional neural network.

The project also integrates **Data Augmentation** tools to enhance model robustness against image variability.

## Project Structure

Py-Image-Classifier/
│
├── dataset/                # <-- Place your original images here
│ 
├── features/               # (Automatically created) Contains processed data
│ ├── augmented/
│ └── ...X_train.npy
│
├── models/                  # (Automatically created) Contains trained models (.joblib)
│
├── results/                 # (Automatically created) Contains reports and charts
│ ├── report...json
│ └── confusion_matrix_...png
│
├── preprocess_images.py        # Script 1
├── extract_features.py         # Script 2
├── train_classifiers.py        # Script 3
├── train_cnn_from_scratch.py   # Script 3 (Alternative)
├── evaluate_results.py         # Script 4
├── app.py                      # Streamlit App
│
├── README.md                 
└── requirements.txt            # Dependencies file


## **Prerequisites**

*   Python 3.9 or higher

## **Installation**

1.  **Download** the files into a folder.

2.  **Open a terminal** at the project root.

3.  **Create a virtual environment** (recommended to isolate dependencies):
    ```bash
    python -m venv venv
    ```

4.  **Activate the virtual environment**:
    *   On Windows:
        ```powershell
        .\venv\Scripts\activate
        ```
   
5.  **Install all required libraries** in a single command:
    ```bash
    pip install -r requirements.txt
    ```

## **Usage Guide**

The pipeline execution follows sequential steps. **The order is crucial.**

### **Step 1: Preprocessing and Data Augmentation**

This script prepares your images and multiplies their number to enrich the dataset.

1.  **Prepare your data**: Place your original images in the `dataset/` folder, organized into subfolders per class.
2.  **Run the script** from the terminal (with the virtual environment activated):
    ```bash
    python preprocess_images.py
    ```
3.  A window will open. **Select the `dataset` folder**.
4.  **Result**: A `features/augmented/` folder is created, containing multiple augmented versions of each original image.

### **Step 2: Training and Evaluation**

There are two approaches in this project:

#### **Option A: Hybrid Learning (Transfer Learning + Classical ML)**
Uses pre-trained models to extract features, then trains classical classifiers.

1.  **Feature Extraction**:
    ```bash
    python extract_features.py
    ```
    *Generates `.npy` files in `features/`.*

2.  **Training Classifiers**:
    ```bash
    # For MobileNetV2
    python train_classifiers.py --features mobilenetv2
    # For ResNet50
    python train_classifiers.py --features resnet50
    ```
    *Generates models in `models/`.*

3.  **Evaluation**:
    ```bash
    # For MobileNetV2
    python evaluate_results.py --features mobilenetv2
    # For ResNet50
    python evaluate_results.py --features resnet50
    ```
    *Generates reports and matrices in `results/`.*

#### **Option B: End-to-End CNN Training (From Scratch)**
Trains a convolutional neural network directly on augmented images.

1.  **Start Training**:
    ```bash
    python train_cnn_from_scratch.py
    ```
2.  **Results**:
    *   Saves the model: `models/cnn_from_scratch.keras`
    *   Saves history: `results/history_cnn_from_scratch.png`
    *   Saves confusion matrix: `results/confusion_matrix_cnn_from_scratch.png`

### **Step 3: Graphical Interface (Web App)**
    
To test your models interactively on new images:
    
1.  **Launch the application**:
    ```bash
    streamlit run app.py
    ```
2.  **Features**:
    *   **Model Selection**: Choose "CNN From Scratch" or "Transfer Learning".
    *   **Upload**: Upload an image (JPG, PNG).
    *   **Prediction**: View the predicted class and confidence level.

## **Script Descriptions**

*   `preprocess_images.py`: Loads raw images, applies augmentations, and saves new images.
*   `extract_features.py`: Extracts features via pre-trained CNNs (MobileNetV2, ResNet50).
*   `train_classifiers.py`: Trains ML classifiers (LDA, kNN, SVM) on extracted features.
*   `train_cnn_from_scratch.py`: Trains a custom CNN directly on images.
*   `evaluate_results.py`: Evaluates hybrid ML models and generates reports.
*   `app.py`: Streamlit Web Application for interactive inference.


## **Author**

*   **TRIBAK Mohamed**
