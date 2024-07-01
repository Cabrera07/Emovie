import pandas as pd
import streamlit as st
import tensorflow as tf
import joblib
import os

# Cargar el archivo CSV
df = pd.read_csv("Data/Processed/processed_movies.csv")

# Redondear los ratings a 1 decimal
df["average_rating"] = df["average_rating"].round(1)

# Añadir columna de información adicional
df["info"] = (
    "Average Rating: " + df["average_rating"].astype(str) + " | Genres: " + df["genres"]
)

# Cargar el modelo entrenado y el MultiLabelBinarizer
model_path = os.path.join('Models', 'movie_recommender_model_4.keras')
mlb_path = os.path.join('Models', 'mlb.joblib')
model = tf.keras.models.load_model(model_path)
mlb = joblib.load(mlb_path)

# Función para convertir rating en estrellas
def rating_to_stars(rating):
    full_star = "★"
    empty_star = "☆"
    return full_star * int(rating) + empty_star * (5 - int(rating))

# Emociones disponibles con emojis
emotion_options = {
    "Joy 😃": "Joy",
    "Fear 😨": "Fear",
    "Disgust 🤢": "Disgust",
    "Sadness 😢": "Sadness",
    "Anger 😠": "Anger",
}

# Función para recomendar películas basadas en el modelo entrenado
def recommend_movies_embedding(emotion1, emotion2, movies, mlb, model, top_n=10):
    emotions = [emotion1, emotion2]
    binarized_emotions = mlb.transform([emotions])
    predicted_rating = model.predict(binarized_emotions)[0][0]
    
    movies['rating_diff'] = abs(movies['average_rating'] - predicted_rating)
    
    recommended_movies = movies.sort_values(by=['rating_diff', 'average_rating'], ascending=[True, False]).head(top_n)
    
    return recommended_movies[['title', 'info', 'average_rating']]

# Incluir CSS para cambiar el tamaño del multiselect, agregar un logo y estilizar los enlaces
st.markdown(
    """
    <style>
    div[data-testid="stMultiSelect"] {
        width: 50%;
    }
    .logo {
        font-size: 50px;
        font-weight: bold;
        color: #ff6347; /* Puedes cambiar el color del logo aquí */
    }
    .tooltip {
        position: relative;
        display: inline-block;
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 220px;
        background-color: black;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%; /* Adjust this to place the tooltip above the link */
        left: 50%;
        margin-left: -110px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    .movie-link {
        display: block;
        color: #ff6347;
        font-weight: bold;
        text-decoration: none;
        padding: 10px;
        margin: 10px 0;
        border-radius: 10px;
        backdrop-filter: blur(10px);
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: background 0.3s, box-shadow 0.3s;
    }
    .movie-link:hover {
        background: rgba(255, 255, 255, 0.3);
        color: #ff6347;
        box-shadow: 0 8px 12px rgba(0, 0, 0, 0.2);
    }
    .stars {
        color: gold;
        font-size: 20px;
        display: inline-block;
        margin-left: 10px;
    }
    </style>
    <div class="logo">E-Movie</div>
    """,
    unsafe_allow_html=True,
)

# Título de la aplicación
st.title("Movie Recommendation Based On Emotions")

# Sección: ¿Cómo te sientes hoy?
st.header("How do you want to feel today?")

selected_emotions = st.multiselect(
    "Select your emotions:", list(emotion_options.keys())
)

# Verificar la cantidad de emociones seleccionadas
if len(selected_emotions) > 2:
    st.error("Please select a maximum of 2 emotions.")
else:
    # Mostrar las recomendaciones de películas
    if st.button("Search Movies"):
        if len(selected_emotions) == 1 or len(selected_emotions) == 2:
            selected_emotions_values = [
                emotion_options[emotion] for emotion in selected_emotions
            ]
            # Para manejar un solo emoción, duplicamos el valor para la predicción
            if len(selected_emotions_values) == 1:
                selected_emotions_values.append(selected_emotions_values[0])
            
            # Obtener recomendaciones de películas utilizando el modelo entrenado
            recommended_movies = recommend_movies_embedding(selected_emotions_values[0], selected_emotions_values[1], df, mlb, model)

            # Mostrar los resultados
            if not recommended_movies.empty:
                st.header("Recommended Movies:")
                for index, row in recommended_movies.iterrows():
                    title = row["title"]
                    info = row["info"]
                    rating = row["average_rating"]
                    stars = rating_to_stars(rating)
                    search_url = f"https://www.google.com/search?q={title}"
                    st.markdown(
                        f"""
                        <div class="tooltip">
                            <a class="movie-link" href="{search_url}" target="_blank">
                                {title} <span class="stars">{stars}</span>
                            </a>
                            <span class="tooltiptext">{info}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.write("No movies found for the selected emotions.")
        else:
            st.write("Please select at least one emotion.")
