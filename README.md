# Limpieza de texto: El arte de la guerra (Sun Tzu)

Proyecto de Procesamiento de Lenguaje Natural. Se limpia y normaliza el libro
"El arte de la guerra" (traduccion de Lionel Giles, 1910, dominio publico,
Project Gutenberg #132) usando spaCy.

## Archivos

- `limpieza.py`: limpieza, normalizacion y lematizacion del texto.
- `descargar_libro.py`: descarga el libro desde Project Gutenberg.
- `arte_de_la_guerra.txt`: libro a procesar.
- `requirements.txt`: dependencias congeladas.

## Proceso de limpieza

1. Se eliminan el encabezado y la licencia de Project Gutenberg.
2. Se normalizan comillas y acentos, y se eliminan URLs, caracteres chinos, notas como [5] y guiones bajos.
3. Se tokeniza y se lematiza con spaCy.
4. Se descartan stopwords, signos de puntuacion y tokens que no sean palabras.
5. El resultado se guarda en `texto_limpio.txt`.

## Uso

    python -m venv venv
    venv\Scripts\activate          (Windows)
    source venv/bin/activate       (Linux / Mac)
    pip install -r requirements.txt
    python descargar_libro.py      (solo si falta el .txt)
    python limpieza.py
