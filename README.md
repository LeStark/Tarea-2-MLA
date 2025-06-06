# Tarea-2
Tarea de aplicaciones de ML

# 🎬 Clasificación de Géneros Cinematográficos con sLDA

Este proyecto implementa un modelo supervisado de tópicos (`sLDA`) para predecir el género de una película a partir de su sinopsis, utilizando técnicas de procesamiento de lenguaje natural y aprendizaje automático.

## 📂 Estructura del proyecto

```
├── Data/                       # Carpeta de datos (no incluida en el repo por peso)
│   ├── movies_metadata.csv
├── utils.py                   # Funciones auxiliares para preprocesamiento y modelado
├── notebook.ipynb             # Notebook principal con flujo de análisis
├── requirements.in            # Lista simple de dependencias
├── requirements.txt           # Lista de dependencias con versiones (generada)
├── .gitignore                 # Archivos a excluir del repositorio
└── README.md
```

## 🔍 Objetivo

Desarrollar un pipeline completo que:

- Preprocese sinopsis con spaCy.
- Genere representaciones vectoriales usando `Supervised Latent Dirichlet Allocation (sLDA)`.
- Clasifique si una película pertenece al género **Drama** (`y_real = 1`) o no (`y_real = 0`).
- Evalúe distintos valores de `k` usando validación cruzada.

## 📊 Dataset

Se utiliza el conjunto de datos [The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset), que contiene información sobre más de 45.000 películas.  
Los archivos muy grandes (`credits.csv`, `ratings.csv`) fueron excluidos del repositorio.

## ⚙️ Instalación

1. Clona el repositorio:
   ```bash
   git clone https://github.com/tu-usuario/tu-repo.git
   cd tu-repo
   ```

2. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

3. Descarga el modelo de spaCy:
   ```bash
   python -m spacy download en_core_web_sm
   ```

## 🚀 Uso

Ejecuta el notebook `notebook.ipynb` para reproducir todo el flujo de análisis:
- Limpieza y preprocesamiento de texto
- Generación de embeddings temáticos
- Validación cruzada para seleccionar el mejor `k`
- Entrenamiento final del modelo sLDA
- Visualización de tópicos y coeficientes

## 🧠 Modelado

- Modelo: `tomotopy.SLDAModel` (modo supervisado binario).
- Entrenamiento con etiquetas `y_real` que identifican películas de género Drama.
- Clasificador auxiliar: regresión logística sobre las distribuciones de tópicos.
- Validación cruzada `k-fold` para encontrar el mejor número de temas (`k`).
