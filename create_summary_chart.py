import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# --- Étape 1 : Entrez vos données ---
# Remplacez les valeurs ci-dessous par les précisions exactes que vous avez obtenues.
data = {
    'Approche': [
        'Deep Learning\n"End-to-End"', 
        'Hybride\n(MobileNetV2)', 
        'Hybride\n(ResNet-50)'
    ],
    'Modèle "Champion"': [
        'CNN "from scratch"', 
        'SVM linéaire', 
        'SVM linéaire'
    ],
    'Précision (Accuracy)': [
        0.9200,  # Précision du modèle "from scratch"
        0.8397,  # Meilleure précision obtenue avec MobileNetV2
        0.9255   # Meilleure précision obtenue avec ResNet-50
    ]
}

# Créer un DataFrame pandas, ce qui est pratique pour la visualisation
df = pd.DataFrame(data)

# Trier les données pour un meilleur visuel (de la meilleure à la moins bonne performance)
df = df.sort_values(by='Précision (Accuracy)', ascending=False)

# --- Étape 2 : Création du graphique ---
plt.style.use('seaborn-v0_8-whitegrid') # Utiliser un style professionnel
fig, ax = plt.subplots(figsize=(10, 6)) # Définir la taille de la figure

# Créer le diagramme en barres avec Seaborn pour un joli rendu
# On utilise une palette de couleurs pour distinguer les barres
palette = ['#4c72b0', '#55a868', '#c44e52']
barplot = sns.barplot(
    x='Approche', 
    y='Précision (Accuracy)', 
    data=df, 
    ax=ax,
    palette=palette
)

# --- Étape 3 : Amélioration du visuel (Titres, Labels, Annotations) ---

# Ajouter le titre principal et les titres des axes
ax.set_title('Graphique Comparatif des Précisions des Modèles', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Approche Méthodologique', fontsize=12, labelpad=15)
ax.set_ylabel('Précision (Accuracy)', fontsize=12, labelpad=15)

# Définir les limites de l'axe Y pour mieux voir les différences (ex: de 75% à 100%)
ax.set_ylim(0.75, 1.0) 

# Formater l'axe Y pour afficher des pourcentages
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

# Ajouter la valeur exacte au-dessus de chaque barre pour plus de clarté
for p in ax.patches:
    ax.annotate(
        f'{p.get_height():.2%}', # Format en pourcentage avec 2 décimales
        (p.get_x() + p.get_width() / 2., p.get_height()), 
        ha='center', 
        va='center', 
        xytext=(0, 9), 
        textcoords='offset points',
        fontsize=12,
        fontweight='bold'
    )

# Ajuster la mise en page pour éviter que les labels se chevauchent
plt.tight_layout()

# --- Étape 4 : Sauvegarder et afficher le graphique ---
output_path = 'results/figure_comparative_accuracy.png'
plt.savefig(output_path, dpi=300) # dpi=300 pour une haute qualité d'image

print(f"Graphique comparatif sauvegardé avec succès dans : {output_path}")

plt.show() # Afficher le graphique à l'écran