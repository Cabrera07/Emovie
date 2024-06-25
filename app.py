import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Conjunto de datos simulado de películas
peliculas = pd.DataFrame({
    'titulo': ['Sueño de fuga', 'El Padrino', 'El Caballero Oscuro', 'Tiempos Violentos', 'El Señor de los Anillos'],
    'descripcion': [
        'Dos hombres encarcelados se unen a lo largo de varios años, encontrando consuelo y eventual redención a través de actos de decencia común.',
        'El envejecido patriarca de una dinastía del crimen organizado transfiere el control de su imperio clandestino a su hijo reacio.',
        'Cuando la amenaza conocida como el Joker emerge de su misterioso pasado, causa estragos y caos en la gente de Gotham.',
        'Las vidas de dos sicarios, un boxeador, la esposa de un gángster y un par de bandidos de cafetería se entrelazan en cuatro historias de violencia y redención.',
        'Un humilde Hobbit de la Comarca y ocho compañeros emprenden un viaje para destruir el poderoso Anillo Único y salvar la Tierra Media del Señor Oscuro Sauron.'
    ]
})

# Vectorizar las descripciones
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matriz = tfidf.fit_transform(peliculas['descripcion'])

# Calcular la matriz de similitud del coseno
similitud_coseno = linear_kernel(tfidf_matriz, tfidf_matriz)

# Función para obtener recomendaciones basadas en el puntaje de similitud del coseno
def obtener_recomendaciones(titulo, similitud_coseno=similitud_coseno):
    # Obtener el índice de la película que coincide con el título
    idx = peliculas[peliculas['titulo'] == titulo].index[0]

    # Obtener los puntajes de similitud de todas las películas con esa película
    puntajes_similitud = list(enumerate(similitud_coseno[idx]))

    # Ordenar las películas basadas en los puntajes de similitud
    puntajes_similitud = sorted(puntajes_similitud, key=lambda x: x[1], reverse=True)

    # Obtener los puntajes de las 3 películas más similares
    puntajes_similitud = puntajes_similitud[1:4]

    # Obtener los índices de las películas
    indices_peliculas = [i[0] for i in puntajes_similitud]

    # Devolver las 3 películas más similares
    return peliculas['titulo'].iloc[indices_peliculas]

# Aplicación Streamlit
st.title('Sistema de Recomendación de Películas')
st.write('Ingresa el título de una película que te guste, y te recomendaremos películas similares.')

titulo_pelicula = st.text_input('Título de la Película')

if titulo_pelicula:
    recomendaciones = obtener_recomendaciones(titulo_pelicula)
    st.write('Recomendaciones:')
    for rec in recomendaciones:
        st.write(rec)
