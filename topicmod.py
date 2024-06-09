import pandas as pd
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

# Cargar el archivo CSV
file_path = r'C:\Universidad\2024-1\Seminario 1\Web Scraping\Topic Modeling\textos_procesados.csv'
data = pd.read_csv(file_path)

# Asegurarse de tener las stopwords en español
nltk.download('stopwords')
stop_words = set(stopwords.words('spanish'))

# Llenar valores nulos en 'texto_limpio' con una cadena vacía
data['texto_limpio'] = data['texto_limpio'].fillna('')

# Preprocesamiento del texto
def preprocess(text):
    return [word for word in text.split() if word not in stop_words]

# Aplicar el preprocesamiento al texto limpio
data['tokens'] = data['texto_limpio'].apply(preprocess)

# Crear el diccionario y el corpus necesarios para LDA
id2word = corpora.Dictionary(data['tokens'])
texts = data['tokens']
corpus = [id2word.doc2bow(text) for text in texts]

# Construir el modelo LDA
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=id2word,
                                            num_topics=5, 
                                            random_state=100,
                                            update_every=1,
                                            chunksize=10,
                                            passes=10,
                                            alpha='auto',
                                            per_word_topics=True)

# Mostrar los tópicos
topics = lda_model.print_topics(num_words=10)
for topic in topics:
    print(topic)

# Asignar el tópico dominante a cada documento
def get_dominant_topic(lda_model, corpus):
    dominant_topics = []
    for i, corp in enumerate(corpus):
        topic_percs, wordid_topics, wordid_phivalues = lda_model[corp]
        dominant_topic = sorted(topic_percs, key=lambda x: x[1], reverse=True)[0][0]
        dominant_topics.append(dominant_topic)
    return dominant_topics

data['dominant_topic'] = get_dominant_topic(lda_model, corpus)

# Mostrar el dataframe con los tópicos asignados
print(data[['Texto', 'texto_limpio', 'dominant_topic']].head())

# Guardar el resultado en un nuevo archivo CSV
output_file_path = r'C:\Universidad\2024-1\Seminario 1\Web Scraping\Topic Modeling\textos_procesados_con_temas.csv'
data.to_csv(output_file_path, index=False)

print('Dataset guardado')
