import pandas as pd
from collections import Counter
import os

def load_data():
    """Cargar los datasets de películas y calificaciones"""
    base_path = os.path.dirname(os.path.abspath(__file__))
    movies_path = os.path.join(base_path, '../Data/Raw/movie.csv')
    ratings_path = os.path.join(base_path, '../Data/Raw/rating.csv')
    
    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)
    return movies, ratings

def preprocess_data(movies, ratings):
    """Preprocesar los datasets"""
    # Eliminar filas con valores nulos
    movies = movies.dropna(subset=['movieId', 'title', 'genres'])
    ratings = ratings.dropna(subset=['movieId', 'rating'])
    
    # Eliminar filas con "(no genres listed)"
    movies = movies[movies['genres'] != "(no genres listed)"]
    
    # Eliminar duplicados
    movies = movies.drop_duplicates(subset=['movieId'])
    ratings = ratings.drop_duplicates(subset=['userId', 'movieId'])
    
    # Eliminar columnas innecesarias
    ratings = ratings.drop(columns=['userId', 'timestamp'])
    
    return movies, ratings

def map_genres_to_emotions(genres):
    """Mapear géneros a emociones"""
    genre_to_emotion = {
        'Drama': 'Sadness',
        'Comedy': 'Amusement',
        'Thriller': 'Suspense',
        'Romance': 'Love',
        'Action': 'Excitement',
        'Crime': 'Anger',
        'Horror': 'Fear',
        'Documentary': 'Curiosity',
        'Adventure': 'Excitement',
        'Sci-Fi': 'Wonder',
        'Mystery': 'Suspense',
        'Fantasy': 'Wonder',
        'War': 'Anger',
        'Children': 'Joy',
        'Musical': 'Joy',
        'Animation': 'Joy',
        'Western': 'Excitement',
        'Film-Noir': 'Mystery'
    }
    genres_list = genres.split('|')
    emotions = [genre_to_emotion.get(genre) for genre in genres_list if genre in genre_to_emotion]
    return emotions

def apply_emotions_mapping(movies):
    """Aplicar la función de mapeo de emociones al dataframe de películas"""
    movies['emotions'] = movies['genres'].apply(map_genres_to_emotions)
    return movies

def calculate_average_ratings(ratings):
    """Calcular el rating promedio para cada película"""
    average_ratings = ratings.groupby('movieId')['rating'].mean().reset_index()
    average_ratings.columns = ['movieId', 'average_rating']
    return average_ratings

def merge_datasets(movies, average_ratings):
    """Unir el rating promedio con el dataframe de películas"""
    movies = pd.merge(movies, average_ratings, on='movieId', how='left')
    return movies

def fill_na_with_global_average(movies, ratings):
    """Rellenar valores NaN con el rating promedio global"""
    global_average_rating = ratings['rating'].mean()
    movies['average_rating'] = movies['average_rating'].fillna(global_average_rating)
    return movies

def save_processed_data(movies):
    """Guardar el dataframe procesado"""
    processed_dir = '../Data/Processed/'
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
    file_path = os.path.join(processed_dir, 'processed_movies.csv')
    if os.path.exists(file_path):
        os.remove(file_path)  # Eliminar el archivo existente antes de guardar el nuevo
    movies.to_csv(file_path, index=False)
    print("El DataFrame procesado ha sido guardado correctamente.")

def verify_data(movies):
    """Verificación final de los datos procesados"""
    print("Verificación de valores NaN en el DataFrame procesado:")
    print(movies.isna().sum())
    print(movies.head())

def main():
    movies, ratings = load_data()
    movies, ratings = preprocess_data(movies, ratings)
    movies = apply_emotions_mapping(movies)
    average_ratings = calculate_average_ratings(ratings)
    movies = merge_datasets(movies, average_ratings)
    movies = fill_na_with_global_average(movies, ratings)
    save_processed_data(movies)
    verify_data(movies)

if __name__ == "__main__":
    main()
