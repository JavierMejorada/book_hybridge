import re
import unicodedata
from collections import Counter

import spacy

RUTA_LIBRO = "arte_de_la_guerra.txt"
RUTA_SALIDA = "texto_limpio.txt"

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])


def leer_libro(ruta):
    with open(ruta, encoding="utf-8") as archivo:
        return archivo.read()


def quitar_encabezado_y_pie(texto):
    inicio = texto.find("*** START OF")
    if inicio != -1:
        texto = texto[texto.find("\n", inicio) + 1:]

    fin = texto.find("*** END OF")
    if fin != -1:
        texto = texto[:fin]

    return texto


def limpiar_parrafo(parrafo):
    parrafo = parrafo.replace("’", "'").replace("‘", "'")
    parrafo = parrafo.replace("“", '"').replace("”", '"')
    parrafo = re.sub(r"https?://\S+", " ", parrafo)
    parrafo = re.sub(r"[\u4e00-\u9fff]+", " ", parrafo)
    parrafo = re.sub(r"\[\d+\]", " ", parrafo)
    parrafo = parrafo.replace("_", " ")
    parrafo = unicodedata.normalize("NFKD", parrafo)
    parrafo = parrafo.encode("ascii", "ignore").decode("ascii")
    parrafo = re.sub(r"\s+", " ", parrafo)
    return parrafo.strip()


def dividir_en_parrafos(texto):
    parrafos = []
    for bloque in re.split(r"\n\s*\n", texto):
        limpio = limpiar_parrafo(bloque)
        if limpio != "":
            parrafos.append(limpio)
    return parrafos


def lematizar(parrafos):
    tokens_limpios = []

    for doc in nlp.pipe(parrafos, batch_size=50):
        for token in doc:
            if token.is_stop or token.is_punct or token.is_space:
                continue
            if not token.is_alpha or len(token.text) < 2:
                continue
            tokens_limpios.append(token.lemma_.lower())

    return tokens_limpios


def main():
    texto = leer_libro(RUTA_LIBRO)
    texto = quitar_encabezado_y_pie(texto)
    parrafos = dividir_en_parrafos(texto)
    tokens = lematizar(parrafos)

    with open(RUTA_SALIDA, "w", encoding="utf-8") as archivo:
        archivo.write(" ".join(tokens))

    print("Parrafos procesados:", len(parrafos))
    print("Tokens limpios:", len(tokens))
    print("Vocabulario unico:", len(set(tokens)))
    print("Palabras mas frecuentes:")
    for palabra, cantidad in Counter(tokens).most_common(15):
        print(f"  {palabra}: {cantidad}")


if __name__ == "__main__":
    main()
