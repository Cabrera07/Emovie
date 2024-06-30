# Importing libraries
import pandas as pd
from collections import Counter
import os

def load_data():
    """Load the movies and ratings datasets"""
    base_path = os.path.dirname(os.path.abspath(__file__))
    movies_path = os.path.join(base_path, '..', 'Data', 'Raw', 'movie.csv')
    ratings_path = os.path.join(base_path, '..', 'Data', 'Raw', 'rating.csv')
    
    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)
    return movies, ratings

def preprocess_data(movies, ratings):
    """Preprocess the datasets"""
    # Remove rows with null values
    movies = movies.dropna(subset=['movieId', 'title', 'genres'])
    ratings = ratings.dropna(subset=['movieId', 'rating'])
    
    # Remove rows with "(no genres listed)"
    movies = movies[movies['genres'] != "(no genres listed)"]
    
    # Remove duplicates
    movies = movies.drop_duplicates(subset=['movieId'])
    ratings = ratings.drop_duplicates(subset=['userId', 'movieId'])
    
    # Drop unnecessary columns
    ratings = ratings.drop(columns=['userId', 'timestamp'])
    
    return movies, ratings

def map_genres_to_emotions(genres):
    """Map genres to emotions"""
    genre_to_emotion = {
        'Drama': 'Sadness',
        'Comedy': 'Joy',
        'Thriller': 'Fear',
        'Romance': 'Joy',
        'Action': 'Anger',
        'Crime': 'Disgust',
        'Horror': 'Fear',
        'Documentary': 'Joy',
        'Adventure': 'Anger',
        'Sci-Fi': 'Fear',
        'Mystery': 'Disgust',
        'Fantasy': 'Joy',
        'War': 'Disgust',
        'Children': 'Joy',
        'Musical': 'Joy',
        'Animation': 'Joy',
        'Western': 'Anger',
        'Film-Noir': 'Sadness'
    }
    genres_list = genres.split('|')
    emotions = [genre_to_emotion.get(genre) for genre in genres_list if genre in genre_to_emotion]
    return emotions

def apply_emotions_mapping(movies):
    """Apply the emotion mapping function to the movies dataframe"""
    movies['emotions'] = movies['genres'].apply(map_genres_to_emotions)
    return movies

def calculate_average_ratings(ratings):
    """Calculate the average rating for each movie"""
    average_ratings = ratings.groupby('movieId')['rating'].mean().reset_index()
    average_ratings.columns = ['movieId', 'average_rating']
    return average_ratings

def merge_datasets(movies, average_ratings):
    """Merge the average rating with the movies dataframe"""
    movies = pd.merge(movies, average_ratings, on='movieId', how='left')
    return movies

def fill_na_with_global_average(movies, ratings):
    """Fill NaN values with the global average rating"""
    global_average_rating = ratings['rating'].mean()
    movies['average_rating'] = movies['average_rating'].fillna(global_average_rating)
    return movies

def save_processed_data(movies):
    """Save the processed dataframe"""
    base_path = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(base_path, '..', 'Data', 'Processed')
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
    file_path = os.path.join(processed_dir, 'processed_movies.csv')
    if os.path.exists(file_path):
        os.remove(file_path)  
    movies.to_csv(file_path, index=False)
    print("The processed DataFrame has been saved successfully.")

def verify_data(movies):
    """Final verification of processed data"""
    print("Verification of NaN values in the processed DataFrame:")
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
