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

| Tema | Palabras Clave Principales                                             | Título Propuesto                     |
|------|------------------------------------------------------------------------|--------------------------------------|
| 0    | film, documentary, world, life, history                                | Documentales y Cine Histórico        |
| 1    | doctor, hospital, patient, mental, disease                             | Salud y Enfermedades Mentales        |
| 2    | murder, police, crime, prison, kill                                    | Crimen y Justicia Penal              |
| 3    | school, student, high, girl, teacher                                   | Vida Escolar y Adolescencia          |
| 4    | team, game, win, player, football                                      | Deportes y Competencias              |
| 5    | war, country, american, story, group                                   | Conflictos Bélicos y Sociedad        |
| 6    | life, story, man, world, young                                         | Existencia, Identidad y Superación   |
| 7    | love, woman, young, fall, husband                                      | Romance y Relaciones de Pareja       |
| 8    | film, movie, story, play, base                                         | Cine y Producción Fílmica            |
| 9    | family, year, old, father, young                                       | Dinámicas Familiares                 |
| 10   | money, work, job, company, business                                    | Trabajo y Ambición Económica         |
| 11   | man, dance, king, street, building                                     | Vida Urbana y Expresión Artística    |
| 12   | mysterious, kill, dead, find, house                                    | Misterio y Terror Doméstico          |
| 13   | friend, find, time, day, night                                         | Amistad y Paso del Tiempo            |
| 14   | man, gang, town, set, battle                                           | Violencia y Lucha Territorial        |
| 15   | earth, island, world, crew, ship                                       | Exploración y Aventuras Fantásticas  |
| 16   | comedy, stand, special, music, star                                    | Comedia y Espectáculo en Vivo        |

