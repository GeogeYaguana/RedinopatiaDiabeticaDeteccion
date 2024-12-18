import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
from PIL import UnidentifiedImageError
import os
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

# Visualizar los resultados
def analisisBrilloLevelRedinopatia():
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

    levels = list(brightness_stats.keys())
    avg_brightness = list(brightness_stats.values())

    sns.barplot(x=levels, y=avg_brightness, palette='magma')
    plt.title('Brillo Promedio por Nivel de Retinopatía')
    plt.xlabel('Nivel de Retinopatía')
    plt.ylabel('Brillo Promedio')
    plt.show()

def summarize_rgb_intensities(labels_csv_path, images_folder):
    """
    Resume las distribuciones de intensidades RGB por nivel de retinopatía.

    Parámetros:
    - labels_csv_path: Ruta al archivo CSV con las etiquetas (debe contener columnas 'image' y 'level').
    - images_folder: Ruta a la carpeta que contiene las imágenes.

    Retorna:
    - DataFrame con estadísticas resumidas para cada nivel y cada canal de color.
    """
    # Cargar las etiquetas
    df = pd.read_csv(labels_csv_path)
    print(df.columns)
        
    # Inicializar una lista para almacenar los resúmenes
    summaries = []
    
    # Iterar sobre cada fila del DataFrame
    for index, row in df.iterrows():
        img_name = row['image']
        level = row['level']
        img_path = os.path.join(images_folder, f"{img_name}.jpeg")
        
        # Abrir la imagen y convertir a RGB
        try:
            img = Image.open(img_path).convert('RGB')
            img_arr = np.array(img)
        except Exception as e:
            print(f"Error al abrir la imagen {img_name}: {e}")
            continue  # Saltar esta imagen y continuar
        
        # Calcular estadísticas para cada canal
        r = img_arr[:, :, 0].flatten()
        g = img_arr[:, :, 1].flatten()
        b = img_arr[:, :, 2].flatten()
        
        summary = {
            'level': level,
            'r_mean': r.mean(),
            'r_median': np.median(r),
            'r_std': r.std(),
            'r_min': r.min(),
            'r_max': r.max(),
            'r_25th': np.percentile(r, 25),
            'r_75th': np.percentile(r, 75),
            
            'g_mean': g.mean(),
            'g_median': np.median(g),
            'g_std': g.std(),
            'g_min': g.min(),
            'g_max': g.max(),
            'g_25th': np.percentile(g, 25),
            'g_75th': np.percentile(g, 75),
            
            'b_mean': b.mean(),
            'b_median': np.median(b),
            'b_std': b.std(),
            'b_min': b.min(),
            'b_max': b.max(),
            'b_25th': np.percentile(b, 25),
            'b_75th': np.percentile(b, 75),
        }
        
        summaries.append(summary)
    
    # Convertir la lista de resúmenes en un DataFrame
    summary_df = pd.DataFrame(summaries)
    
    # Agrupar por nivel y calcular las estadísticas agregadas (opcional)
    grouped_summary = summary_df.groupby('level').agg({
        'r_mean': 'mean',
        'r_median': 'mean',
        'r_std': 'mean',
        'r_min': 'min',
        'r_max': 'max',
        'r_25th': 'mean',
        'r_75th': 'mean',
        
        'g_mean': 'mean',
        'g_median': 'mean',
        'g_std': 'mean',
        'g_min': 'min',
        'g_max': 'max',
        'g_25th': 'mean',
        'g_75th': 'mean',
        
        'b_mean': 'mean',
        'b_median': 'mean',
        'b_std': 'mean',
        'b_min': 'min',
        'b_max': 'max',
        'b_25th': 'mean',
        'b_75th': 'mean',
    }).reset_index()
    
    return grouped_summary
result = summarize_rgb_intensities('ejemplos/labels.csv', 'ejemplos')
print(result)