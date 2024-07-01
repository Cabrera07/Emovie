# E-Movie: Sistema de Recomendación de Películas Basado en Emociones

## Descripción

E-Movie es un sistema de recomendación de películas que utiliza un mapeo de géneros a emociones para sugerir contenido relevante a los usuarios. Al aprovechar técnicas de aprendizaje profundo (deep learning), E-Movie ofrece recomendaciones personalizadas basadas en las emociones que los géneros de las películas pueden evocar. Las emociones se definen mediante un mapeo de géneros a emociones.

## Objetivo

El objetivo de E-Movie es mejorar la experiencia de los usuarios al recomendarles películas que coincidan con su estado emocional actual o deseado. Este sistema busca:

- Proporcionar recomendaciones precisas basadas en un modelo de red neuronal profunda.
  
- Ofrecer una plataforma intuitiva y fácil de usar para explorar nuevas películas.
  
- Mejorar continuamente el algoritmo de recomendación mediante la incorporación de retroalimentación de los usuarios.

## Estructura del Proyecto

El proyecto E-Movie está organizado de la siguiente manera:

```plaintext
Emovie/
│
├── Data/
│   ├── Processed/
│   │   └── processed_movies.csv
│   ├── Raw/
│   │   ├── movie.csv
│   │   └── rating.csv
│
├── Models/
│   ├── mlb.joblib
│   ├── movie_recommender_model_2.keras
│   ├── movie_recommender_model_3.keras
│   ├── movie_recommender_model_4.keras
│   └── movie_recommender_model.keras
│
├── Notebooks/
│   ├── data_preprocessing.ipynb
│   ├── library.ipynb
│   ├── model_training_1.ipynb
│   ├── model_training_2.ipynb
│   ├── model_training_3.ipynb
│   └── model_training_final.ipynb
│
├── src/
│   ├── data_processing.py
│   ├── train_model.py
│
├── .gitignore
├── app.py
└── requirements.txt
```

## Requisitos

- Python 3.9.7

## Instalación

Para instalar y ejecutar el proyecto eMovie, sigue estos pasos:

1. Clona el repositorio:

    ```bash
    git clone https://github.com/Cabrera07/Emovie.git
    ```

2. Crea un entorno virtual y actívalo:
  
    ```bash
    python -m venv env
    env\Scripts\activate  # En Windows usa `source env/bin/activate `
    ```

3. Instala las dependencias:

    ```bash
    pip install -r requirements.txt
    ```

## Datos

Debido al tamaño de los archivos **`rating.csv`** y **`movie.csv`**, no se incluyen directamente en el repositorio. Puedes descargar los archivos desde los siguientes enlaces y colocarlos en la carpeta **`Data/Raw/`**:

[Descargar Archivos](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset?resource=download&select=rating.csv)

## Uso

1. Procesamiento de datos:

    ```bash
    python src/data_processing.py
    ```

2. Entrenamiento del modelo:
  
    ```bash
    python src/train_model.py
    ```

3. Ejecución de la aplicación Streamlit:

    ```bash
    streamlit run app.py
    ```

## Flujo de Trabajo con Ramas

En el proyecto eMovie, utilizamos un flujo de trabajo basado en ramas para asegurar una colaboración efectiva y organizada. A continuación, se describe el flujo de trabajo:

- **main**: La rama **`main`** contiene el código estable y listo para producción. Todos los cambios en esta rama deben ser revisados y aprobados a través de Pull Requests.
  
- **develop**: La rama **`develop`** es donde se integran las características que están listas para ser probadas antes de pasar a **`main`**. Es la rama base para cualquier desarrollo.
  
- **feature/***: Para cada nueva funcionalidad o mejora, se debe crear una rama feature a partir de **`develop`**. Una vez completada la funcionalidad, se realiza un Pull Request para integrar los cambios en **`develop`**.

### Pasos para Contribuir

1. Haz un fork del repositorio.

2. Crea una nueva rama a partir de develop:
  
    ```bash
    git switch -c feature/new-functionality develop
    ```

3. Realiza tus cambios y haz commit:

    ```bash
    git add .
    git commit -am 'feat: add new functionality'
    ```

4. Sube los cambios a tu rama:

    ```bash
    git push origin feature/new-functionality
    ```

5. Crea un Pull Request hacia la rama **`develop`** y describe los cambios realizados.

### Contribución

Si deseas contribuir a eMovie, por favor sigue los pasos mencionados en la sección "Flujo de Trabajo con Ramas".
