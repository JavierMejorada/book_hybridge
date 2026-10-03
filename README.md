# Procesamiento de texto: El arte de la guerra (Sun Tzu)

Proyecto de Procesamiento de Lenguaje Natural con la traduccion de Lionel Giles
(1910, dominio publico, Project Gutenberg #132).

## Archivos

- `limpieza.py`: checkpoint 2. Limpieza, normalizacion y lematizacion con spaCy.
- `vectorizacion.py`: checkpoint 3. Bag-of-Words, TF-IDF y visualizacion 3D con PCA.
- `descargar_libro.py`: descarga el libro desde Project Gutenberg.
- `arte_de_la_guerra.txt`: libro a procesar.
- `requirements.txt`: dependencias congeladas.

## Vectorizacion

1. Se limpia el libro y se separa en oraciones con `doc.sents`.
2. Cada oracion se lematiza (sin stopwords ni puntuacion) y forma el corpus.
3. `CountVectorizer` genera la matriz Bag-of-Words.
4. `TfidfVectorizer` genera la matriz TF-IDF.
5. Se comparan las palabras mas frecuentes contra las de mayor TF-IDF y se mide la sparsity.
6. Con PCA se reducen a 3 dimensiones las 40 palabras mas frecuentes y se grafican ambas representaciones.

Salidas: `corpus_lematizado.txt` y `vectorizacion_3d.png`.

## Uso

    python -m venv venv
    venv\Scripts\activate          (Windows)
    source venv/bin/activate       (Linux / Mac)
    pip install -r requirements.txt
    python descargar_libro.py      (solo si falta el .txt)
    python limpieza.py
    python vectorizacion.py
