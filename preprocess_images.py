import os
import cv2
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm
import albumentations as A


AUGMENTATION_PIPELINES = {
    "flip_horizontal": A.Compose([
        A.Resize(height=224, width=224),
        A.HorizontalFlip(p=1.0),
    ]),
    "rotation": A.Compose([
        A.Resize(height=224, width=224),
        A.Rotate(limit=15, p=1.0, border_mode=cv2.BORDER_CONSTANT),
    ]),
    "zoom_leger": A.Compose([
        A.Resize(height=224, width=224),
        A.ShiftScaleRotate(shift_limit=0, scale_limit=0.2, rotate_limit=0, p=1.0, border_mode=cv2.BORDER_CONSTANT),
    ]),
    
    "flou": A.Compose([
        A.Resize(height=224, width=224),
        A.GaussianBlur(blur_limit=(3, 7), p=1.0),
    ]),

    "luminosite_contraste": A.Compose([
        A.Resize(height=224, width=224),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
    ]),
    # Ce pipeline applique une combinaison aléatoire de plusieurs transformations.
    "combine_aleatoire": A.Compose([
        A.Resize(height=224, width=224),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, p=0.5, border_mode=cv2.BORDER_CONSTANT),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=0, p=0.5, border_mode=cv2.BORDER_CONSTANT),
    ])
}

def preprocess_and_augment_multiple():
    """
    Fonction principale qui génère plusieurs augmentations pour chaque image.
    """
    root = tk.Tk()
    root.withdraw()
    input_dir = filedialog.askdirectory(title="Sélectionnez le dossier racine de votre dataset")
    
    if not input_dir:
        print("Aucun dossier n'a été sélectionné. Le programme va s'arrêter.")
        return

    output_base_dir = "features"
    output_augmented_dir = os.path.join(output_base_dir, "augmented")
    os.makedirs(output_augmented_dir, exist_ok=True)
    
    print("-" * 50)
    print(f"Dossier source : {input_dir}")
    print(f"Dossier de destination : {output_augmented_dir}")
    print(f"Nombre d'augmentations par image : {len(AUGMENTATION_PIPELINES)}")
    print("-" * 50)

    class_names = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    
    if not class_names:
        print(f"Erreur : Aucun sous-dossier (classe) trouvé dans {input_dir}.")
        return

    for class_name in class_names:
        input_class_dir = os.path.join(input_dir, class_name)
        output_class_dir = os.path.join(output_augmented_dir, class_name)
        os.makedirs(output_class_dir, exist_ok=True)
        
        image_files = [f for f in os.listdir(input_class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        print(f"\nTraitement de la classe '{class_name}'...")
        
        for image_name in tqdm(image_files, desc=f"Classe {class_name}"):
            image_path = os.path.join(input_class_dir, image_name)
            
            image = cv2.imread(image_path)
            
            if image is None:
                print(f"\nAttention : Impossible de charger l'image {image_path}, elle sera ignorée.")
                continue
            
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            base_name, ext = os.path.splitext(image_name)

            
            for aug_name, transform in AUGMENTATION_PIPELINES.items():
                
                # Appliquer la transformation
                augmented = transform(image=image_rgb)
                augmented_image_rgb = augmented['image']
                
                # Reconvertir en BGR pour la sauvegarde
                augmented_image_bgr = cv2.cvtColor(augmented_image_rgb, cv2.COLOR_RGB2BGR)

                # Créer un nom de fichier unique pour chaque augmentation
                output_filename = f"{base_name}_aug_{aug_name}{ext}"
                output_path = os.path.join(output_class_dir, output_filename)
                
                # Sauvegarder la nouvelle image
                cv2.imwrite(output_path, augmented_image_bgr)

    print("\n" + "=" * 50)
    print("Prétraitement et augmentation multiple terminés avec succès !")
    print(f"Le nombre d'images a été multiplié par {len(AUGMENTATION_PIPELINES)}.")
    print(f"Les images sont prêtes dans le dossier '{output_augmented_dir}'.")
    print("Vous pouvez maintenant lancer le script 'extract_features.py'.")
    print("=" * 50)

if __name__ == "__main__":
    preprocess_and_augment_multiple()