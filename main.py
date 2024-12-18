import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
from PIL import UnidentifiedImageError

# Analisis Explorativo de Datos
df = pd.read_csv('ejemplos/labels.csv')
print(df.head())
## Ejemplos por clases
class_counts = df['level'].value_counts().sort_index()
print(class_counts)
## Distribucion por clases
sns.countplot(x='level', data=df, palette='viridis')
plt.title('Distribución de las Clases')
plt.xlabel('Nivel de Retinopatía')
plt.ylabel('Cantidad de Imágenes')
plt.show() ## abra que realizar data aumentation para las clases con menor cantidad de imagenes.

#Analisis de las imagenes
ejemplos_images = df['image'].tolist()
def analisisAleatorioImagenes():
    for img_name in ejemplos_images:
        img_path = f'ejemplos/{img_name}.jpeg'
        try:
            img = Image.open(img_path)
            plt.imshow(img)
            plt.title(f'Imagen: {img_name}, Nivel: {df[df["image"] == img_name]["level"].values[0]}')
            plt.axis('off')
            plt.show()
        except Exception as e:
            print(f"Error al abrir la imagen {img_name}: {e}")

widths = []
heights = []
def analisisUniformidadImagenes():
    for img_name in ejemplos_images:
        img_path = f'ejemplos/{img_name}.jpeg'
        img = Image.open(img_path)
        w, h = img.size
        widths.append(w)
        heights.append(h)

    print(f"Ancho promedio: {sum(widths)/len(widths)}")
    print(f"Alto promedio: {sum(heights)/len(heights)}")
    ## Hay variabilidad significativa

def imagenesCorruptasCount():
    corrupt_count = 0
    for img_name in ejemplos_images:
        img_path = f'ejemplos/{img_name}.jpeg'
        try:
            img = Image.open(img_path)
            img.verify()  # Verifica la integridad de la imagen
        except (IOError, UnidentifiedImageError):
            corrupt_count += 1
            print(f"Imagen corrupta detectada: {img_name}")

    print(f"Total de imágenes corruptas: {corrupt_count}")
def analisisBrilloIntensidadRGB():
    for img_name in ejemplos_images:
        img_path = f'ejemplos/{img_name}.jpeg'
        img = Image.open(img_path).convert('RGB')  # Asegura que la imagen esté en RGB
        img_arr = np.array(img)
        avg_intensity = img_arr.mean()
        print(f"Imagen: {img_name}, Brillo Promedio: {avg_intensity:.2f}")
        
        # Histograma de intensidades para el canal rojo
        plt.hist(img_arr[:, :, 0].ravel(), bins=50, alpha=0.5, label='Rojo')
        plt.hist(img_arr[:, :, 1].ravel(), bins=50, alpha=0.5, label='Verde')
        plt.hist(img_arr[:, :, 2].ravel(), bins=50, alpha=0.5, label='Azul')
        plt.title(f"Histograma de Intensidades para {img_name}")
        plt.xlabel('Intensidad de Píxel')
        plt.ylabel('Frecuencia')
        plt.legend()
        plt.show()

# Agrupar por nivel y calcular el brillo promedio
brightness_stats = {}

for level in df['level'].unique():
    subset = df[df['level'] == level]
    brightness = []
    for img_name in subset['image']:
        img_path = f'ejemplos/{img_name}.jpeg'
        img = Image.open(img_path).convert('RGB')
        img_arr = np.array(img)
        brightness.append(img_arr.mean())
    brightness_stats[level] = np.mean(brightness)

# Visualizar los resultados
def analisisBrilloLevelRedinopatia():
    levels = list(brightness_stats.keys())
    avg_brightness = list(brightness_stats.values())

    sns.barplot(x=levels, y=avg_brightness, palette='magma')
    plt.title('Brillo Promedio por Nivel de Retinopatía')
    plt.xlabel('Nivel de Retinopatía')
    plt.ylabel('Brillo Promedio')
    plt.show()

