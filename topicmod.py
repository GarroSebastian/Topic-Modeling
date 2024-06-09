import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# Definir la ruta del archivo de entrada y salida
input_file_path = r'C:\Universidad\2024-1\Seminario 1\Web Scraping\Topic Modeling\textos_procesados.csv'
output_file_path = r'C:\Universidad\2024-1\Seminario 1\Web Scraping\Topic Modeling\textos_procesados_con_temas.csv'

# Cargar el dataset
data = pd.read_csv(input_file_path)

# Preprocesamiento adicional (si es necesario)
data['texto_limpio'] = data['texto_limpio'].astype(str)

# Vectorización del texto
stop_words = stopwords.words('spanish')
vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words=stop_words)
text_vectorized = vectorizer.fit_transform(data['texto_limpio'])

# Construcción del modelo LDA
lda = LatentDirichletAllocation(n_components=9, random_state=42)
lda.fit(text_vectorized)

# Asignación de temas dominantes
dominant_topic = np.argmax(lda.transform(text_vectorized), axis=1)
data['dominant_topic'] = dominant_topic

# Definición de categorías de aspectos para smartphones en español
aspect_categories = {
    0: 'Calidad',
    1: 'Cámara',
    2: 'Batería',
    3: 'Rapidez',
    4: 'Diseño',
    5: 'Pantalla',
    6: 'Software',
    7: 'Precio',
    8: 'Accesorios'
}

data['category_aspect'] = data['dominant_topic'].map(aspect_categories)

# Seleccionar y renombrar las columnas necesarias
data = data[['Producto','Fecha','texto_limpio', 'CalificaciÃ³n', 'dominant_topic', 'category_aspect']]
data.rename(columns={'Producto':'producto', 'Fecha':'fecha', 'texto_limpio': 'texto_limpio_reseña', 'CalificaciÃ³n': 'calificacion'}, inplace=True)

# Guardar el dataset con las nuevas columnas
data.to_csv(output_file_path, index=False)

print("Dataset guardado con éxito en la ruta especificada.")
