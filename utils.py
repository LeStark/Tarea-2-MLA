
import os        # Operaciones del sistema de archivos
import shutil    # Manipulación de archivos y directorios
import ast       # Evaluación segura de literales de Python (e.g., strings con listas/diccionarios)
import re        # Expresiones regulares

import pandas as pd       # Manipulación de estructuras tipo DataFrame
import numpy as np        # Operaciones matemáticas y manejo de arrays

import matplotlib.pyplot as plt  # Gráficos

import spacy        # Procesamiento de lenguaje natural (tokenización, lematización, POS tagging)

from sklearn.linear_model import LogisticRegression         # Clasificador lineal
from sklearn.metrics import accuracy_score                  # Métrica de desempeño
from sklearn.model_selection import KFold                   # Validación cruzada

import tomotopy as tp     # Modelos de tópicos probabilísticos (incl. sLDA)

import kagglehub          # Descarga datasets desde Kaggle con autenticación


def importar_datos_kaggle():
    """
    Verifica si los datos están en la carpeta 'Data' y, si no, los descarga desde Kaggle.

    Esta función verifica si la carpeta `Data` (ubicada en el directorio actual del notebook)
    contiene archivos. Si no existen archivos, descarga el dataset desde Kaggle utilizando
    la biblioteca `kagglehub`, lo mueve a la carpeta `Data` y asegura que los datos estén
    organizados en una ubicación relativa al notebook.

    Pasos realizados por la función:
    1. Comprueba si la carpeta `Data` existe y contiene archivos.
    2. Si no existe o está vacía:
        - Descarga el dataset desde Kaggle.
        - Crea la carpeta `Data` si no existe.
        - Mueve los archivos descargados a la carpeta `Data`.
    3. Si los archivos ya están en `Data`, no realiza ninguna acción.

    Requisitos:
        - Tener instalada y configurada la biblioteca `kagglehub` con las credenciales de Kaggle.
        - Tener permisos de escritura en el directorio actual del notebook.

    Raises:
        FileNotFoundError: Si no se puede encontrar el dataset descargado en la ubicación predeterminada.
        OSError: Si ocurre un error al mover los archivos o crear la carpeta `Data`.

    Examples:
        >>> importar_datos_kaggle()
        Los archivos ya están descargados en la carpeta 'Data'.

    """
    # Ruta relativa a la carpeta del notebook
    data_dir = os.path.join(os.getcwd(), "Data")  # Esto crea la ruta relativa ./Data

    # Verifica si la carpeta existe y contiene archivos
    if not os.path.exists(data_dir) or len(os.listdir(data_dir)) == 0:
        print("No se encontraron archivos en la carpeta 'Data'. Descargando dataset...")
        
        # Descarga el dataset en la ubicación predeterminada
        default_path = kagglehub.dataset_download("rounakbanik/the-movies-dataset")
        
        # Crea la carpeta 'Data' si no existe
        os.makedirs(data_dir, exist_ok=True)
        
        # Mueve los archivos descargados a la carpeta 'Data'
        for file_name in os.listdir(default_path):
            shutil.move(os.path.join(default_path, file_name), data_dir)
        
        print("Dataset descargado y movido a:", data_dir)
    else:
        print("Los archivos ya están descargados en la carpeta 'Data'.")

import pandas as pd
import ast

def generar_datos_entrenamiento():
    """
    Carga y prepara un subconjunto limpio y estructurado del dataset 'The Movies Dataset' 
    para tareas de procesamiento de texto y clasificación de género.

    Returns:
        tuple: 
            - pd.DataFrame: DataFrame `movies` con las columnas 'title', 'overview', 'genre_id', 'genre_name'.
            - pd.DataFrame: DataFrame `tabla_generos` con géneros únicos ('genre_id', 'genre_name').
    """
    importar_datos_kaggle()  # Debe asegurar que el archivo 'movies_metadata.csv' esté disponible

    ruta_relativa = "Data/movies_metadata.csv"
    movies_original = pd.read_csv(ruta_relativa, low_memory=False)

    columnas_relevantes = ['title', 'overview', 'genres']
    movies_original = movies_original[columnas_relevantes]

    movies_filtrado = movies_original.dropna(subset=['overview'])
    movies_filtrado = movies_filtrado[movies_filtrado['overview'].str.strip() != '']

    movies_filtrado['genres'] = movies_filtrado['genres'].apply(
        lambda x: ast.literal_eval(x) if pd.notnull(x) and x != '[]' else []
    )

    movies = movies_filtrado[movies_filtrado['genres'].apply(lambda x: len(x) == 1)].copy()

    movies['genre_id'] = movies['genres'].apply(lambda x: x[0]['id'])
    movies['genre_name'] = movies['genres'].apply(lambda x: x[0]['name'])

    movies = movies.drop(columns=['genres'])

    # Crear tabla de géneros únicos
    tabla_generos = movies[['genre_id', 'genre_name']].drop_duplicates().sort_values('genre_id').reset_index(drop=True)

    return movies, tabla_generos

def preprocesar_texto(texto):
    """
    Función genérica que realiza el preprocesamiento de texto para modelos LDA en español.
    Incluye POS tagging, lematización y filtrado según mejores prácticas.
    Devuelve una lista de tokens procesados.
    """

    nlp = spacy.load("en_core_web_sm")

    # 1. Eliminación de URLs y correos electrónicos
    texto = re.sub(r'http\S+', '', texto)
    texto = re.sub(r'\S+@\S+', '', texto)

    # 2. Normalización de espacios y caracteres especiales
    texto = re.sub(r'\s+', ' ', texto).strip().lower()

    # 3. Procesamiento con SpaCy (tokenización, POS tagging, lematización)
    doc = nlp(texto)

    tokens_procesados = []

    for token in doc:
        # 4. Filtrar stopwords, puntuación y números
        if token.is_stop or token.is_punct or token.like_num:
            continue

        # 5. Filtrar por POS: mantener sustantivos, adjetivos y verbos en infinitivo
        if token.pos_ not in ['NOUN', 'ADJ', 'VERB']:
            continue

        # 6. Lematización y limpieza final
        lemma = token.lemma_.lower().strip()

        # 7. Filtrar lemas cortos y caracteres no deseados
        if len(lemma) < 3 or not re.match(r'^[a-záéíóúñü]+$', lemma):
            continue

        tokens_procesados.append(lemma)

    return tokens_procesados


# Función de limpieza por palabra
def limpiar_palabra(palabra):
    return re.sub(r"[^\w\s]", "", palabra)  # elimina cualquier carácter que no sea letra, número o espacio

def graficar_frecuencia_generos(movies,columna_genero='genre_name'):
    """
    Genera un gráfico de barras horizontal que muestra la distribución de películas por género.
    """
    # Contar frecuencia de géneros
    conteo_generos = movies[columna_genero].value_counts()
    # Crear gráfico de barras horizontal
    plt.figure(figsize=(10, 6))
    conteo_generos.plot(kind='barh', color='skyblue')
    plt.xlabel('Número de películas')
    plt.ylabel('Género')
    plt.title('Distribución de películas por género')
    plt.gca().invert_yaxis()  # Para que el más frecuente quede arriba
    plt.tight_layout()
    plt.show()

def validar_kfold_slda(k, documentos, etiquetas, folds=5, iteraciones=1000):
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    scores = []

    documentos = np.array(documentos, dtype=object)
    etiquetas = np.array(etiquetas)

    for train_idx, test_idx in kf.split(documentos):
        docs_train = documentos[train_idx]
        y_train = etiquetas[train_idx]
        docs_test = documentos[test_idx]
        y_test = etiquetas[test_idx]

        modelo = tp.SLDAModel(k=k, seed=42, vars='b')

        for i in range(len(docs_train)):
            modelo.add_doc(words=docs_train[i], y=[y_train[i]])

        modelo.train(iteraciones)

        # Obtener representaciones para train y test
        X_train = [doc.get_topic_dist() for doc in modelo.docs]
        X_test = [modelo.infer(list(d))[0] for d in docs_test]
        clf = LogisticRegression(max_iter=500)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        scores.append(accuracy_score(y_test, y_pred))

    return np.mean(scores)


def validar_kfold_slda_1(k, documentos, etiquetas, folds=5, iteraciones=100):
    """
    Realiza validación cruzada con k folds para un modelo sLDA con tomotopy.
    
    Args:
        k (int): Número de tópicos para el modelo sLDA.
        documentos (list of list of str): Lista de documentos, cada uno como lista de tokens.
        etiquetas (list of int): Lista de etiquetas binarias (0 o 1).
        folds (int): Número de folds. Default = 5.
        iteraciones (int): Iteraciones de entrenamiento para el modelo sLDA. Default = 100.
        
    Returns:
        float: Accuracy promedio en los folds.
    """
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    scores = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(documentos), 1):
        docs_train = [documentos[i] for i in train_idx]
        y_train = [etiquetas[i] for i in train_idx]
        docs_test = [documentos[i] for i in test_idx]
        y_test = [etiquetas[i] for i in test_idx]

        modelo = tp.SLDAModel(k=k, seed=42, vars='b')

        docs_entrenados = []
        etiquetas_entrenadas = []

        for doc, y in zip(docs_train, y_train):
            palabras = [token.strip() for token in doc if token.strip()]
            if not palabras:
                continue
            try:
                modelo.add_doc(words=palabras, y=[int(y)])
                docs_entrenados.append(palabras)
                etiquetas_entrenadas.append(y)
            except Exception as e:
                print(f"Error al agregar documento: {e}")

        print(f"Entrenando modelo con {len(docs_entrenados)} documentos (fold {fold})")
        modelo.train(iteraciones)

        # Representaciones de entrenamiento
        X_train = [doc.get_topic_dist() for doc in modelo.docs]

        # Inferir representación para documentos de prueba
        X_test = []
        y_test_filtrado = []

        for doc, y in zip(docs_test, y_test):
            palabras = [token.strip() for token in doc if token.strip()]
            if not palabras:
                continue
            try:
                inf_doc = modelo.make_doc(palabras)
                dist = modelo.infer(inf_doc)[0]
                X_test.append(dist)
                y_test_filtrado.append(y)
            except Exception as e:
                print(f"Error al inferir doc de prueba: {e}")

        # Clasificador
        clf = LogisticRegression(max_iter=500)
        clf.fit(X_train, etiquetas_entrenadas)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test_filtrado, y_pred)
        scores.append(acc)
        print(f"Fold {fold} accuracy: {acc:.4f}")

    return np.mean(scores)


def graficar_todos_los_temas_en_una_figura(modelo, num_temas=17, top_n=10):
    cols = 4
    rows = (num_temas + cols - 1) // cols  # Cálculo para ajustar la última fila si es impar

    fig, axes = plt.subplots(rows, cols, figsize=(20, 4 * rows))
    axes = axes.flatten()

    for k in range(num_temas):
        palabras, pesos = zip(*modelo.get_topic_words(k, top_n=top_n))
        axes[k].barh(palabras[::-1], pesos[::-1])
        axes[k].set_title(f"Tema #{k}")
        axes[k].set_xlabel("Peso")
        axes[k].invert_yaxis()

    # Oculta los ejes que no se usan si hay menos de 20 temas
    for i in range(num_temas, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()
