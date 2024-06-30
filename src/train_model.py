# Importing libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.regularizers import l2

# Function to load data from a CSV file
def load_data(movies_path):
    movies = pd.read_csv(movies_path)
    return movies

# Function to preprocess the emotions data in the movies dataset
def preprocess_emotions(movies):
    movies['emotions_list'] = movies['emotions'].apply(lambda x: eval(x))
    mlb = MultiLabelBinarizer()
    emotions_binarized = mlb.fit_transform(movies['emotions_list'])
    return movies, emotions_binarized, mlb

# Function to split the data into training and testing sets
def split_data(emotions_binarized, ratings):
    return train_test_split(emotions_binarized, ratings, test_size=0.2, random_state=42)

# Function to create the neural network model
def create_model(input_shape):
    input_layer = Input(shape=input_shape)
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(input_layer)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.3)(x)
    output_layer = Dense(1, activation='linear')(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Function to train the model
def train_model(model, X_train, y_train, X_test, y_test):
    history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test), batch_size=32)
    return history

# Function to save the trained model
def save_model(model, model_path):
    model.save(model_path)

# Function to evaluate the model on the test set
def evaluate_model(model, X_test, y_test):
    loss, mae = model.evaluate(X_test, y_test)
    print(f'Model Loss: {loss}, Model MAE: {mae}')
    return loss, mae

# Function to recommend movies based on the input emotions
def recommend_movies_embedding(emotion1, emotion2, movies, mlb, model, top_n=10):
    emotions = [emotion1, emotion2]
    binarized_emotions = mlb.transform([emotions])
    predicted_rating = model.predict(binarized_emotions)[0][0]
    
    movies['rating_diff'] = abs(movies['average_rating'] - predicted_rating)
    
    recommended_movies = movies.sort_values(by=['rating_diff', 'average_rating'], ascending=[True, False]).head(top_n)
    
    return recommended_movies[['title', 'average_rating']]

# Define file paths
movies_path = os.path.join('..', 'Data', 'Processed', 'processed_movies.csv')
model_path = os.path.join('..', 'Models', 'movie_recommender_model_4.keras')

if __name__ == "__main__":
    movies = load_data(movies_path)
    movies, emotions_binarized, mlb = preprocess_emotions(movies)
    X_train, X_test, y_train, y_test = split_data(emotions_binarized, movies['average_rating'])
    model = create_model((emotions_binarized.shape[1],))
    history = train_model(model, X_train, y_train, X_test, y_test)
    save_model(model, model_path)
    evaluate_model(model, X_test, y_test)
    
    # Example recommendation
    print("Recomendaciones para las emociones 'Joy' y 'Fear':")
    print(recommend_movies_embedding('Joy', 'Fear', movies, mlb, model))
