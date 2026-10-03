import numpy as np
import spacy
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from limpieza import leer_libro, quitar_encabezado_y_pie, dividir_en_parrafos

RUTA_LIBRO = "arte_de_la_guerra.txt"
RUTA_CORPUS = "corpus_lematizado.txt"
RUTA_GRAFICA = "vectorizacion_3d.png"
PALABRAS_A_GRAFICAR = 40

nlp = spacy.load("en_core_web_sm", disable=["ner"])


def obtener_corpus_lematizado(parrafos):
    corpus_lematizado = []

    for doc in nlp.pipe(parrafos, batch_size=50):
        for oracion in doc.sents:
            lemas_oracion = [
                token.lemma_.lower()
                for token in oracion
                if not token.is_punct and not token.is_space and not token.is_stop
                and token.is_alpha and len(token.text) > 1
            ]
            if lemas_oracion:
                corpus_lematizado.append(" ".join(lemas_oracion))

    return corpus_lematizado


def mostrar_top(nombre, valores, vocabulario, n=10):
    posiciones = np.argsort(valores)[::-1][:n]
    print(nombre)
    for i in posiciones:
        print(f"  {vocabulario[i]}: {valores[i]:.3f}")


def graficar_palabras_3d(ax, matriz, vocabulario, titulo, color_puntos):
    matriz_palabras = matriz.T

    pca = PCA(n_components=3)
    coords = pca.fit_transform(matriz_palabras.toarray())

    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]

    ax.scatter(x, y, z, c=color_puntos, s=80, edgecolors="k", alpha=0.8, depthshade=True)

    for i, palabra in enumerate(vocabulario):
        ax.text(x[i], y[i], z[i] + 0.1, palabra, fontsize=9)

    ax.set_title(titulo)
    ax.set_xlabel("Comp. Principal 1")
    ax.set_ylabel("Comp. Principal 2")
    ax.set_zlabel("Comp. Principal 3")


def main():
    texto = leer_libro(RUTA_LIBRO)
    texto = quitar_encabezado_y_pie(texto)
    parrafos = dividir_en_parrafos(texto)
    corpus_lematizado = obtener_corpus_lematizado(parrafos)

    with open(RUTA_CORPUS, "w", encoding="utf-8") as archivo:
        archivo.write("\n".join(corpus_lematizado))

    print("Total de oraciones procesadas:", len(corpus_lematizado))

    bow_vectorizer = CountVectorizer()
    X_bow = bow_vectorizer.fit_transform(corpus_lematizado)
    vocabulario = bow_vectorizer.get_feature_names_out()

    tfidf_vectorizer = TfidfVectorizer()
    X_tfidf = tfidf_vectorizer.fit_transform(corpus_lematizado)

    print("Forma de la matriz BoW:", X_bow.shape)
    print("Forma de la matriz TF-IDF:", X_tfidf.shape)

    celdas_totales = X_bow.shape[0] * X_bow.shape[1]
    ceros = 1 - X_bow.nnz / celdas_totales
    print(f"Porcentaje de ceros (sparsity): {ceros * 100:.2f}%")

    conteos = np.asarray(X_bow.sum(axis=0)).ravel()
    promedios_tfidf = np.asarray(X_tfidf.mean(axis=0)).ravel()

    print()
    mostrar_top("Palabras con mas conteos (BoW):", conteos, vocabulario)
    print()
    mostrar_top("Palabras con mayor TF-IDF promedio:", promedios_tfidf, vocabulario)

    posiciones = np.argsort(conteos)[::-1][:PALABRAS_A_GRAFICAR]
    vocab_grafica = vocabulario[posiciones]

    fig = plt.figure(figsize=(18, 8))

    ax1 = fig.add_subplot(121, projection="3d")
    graficar_palabras_3d(ax1, X_bow[:, posiciones], vocab_grafica,
                         "Espacio BoW 3D (Conteos)", "orange")

    ax2 = fig.add_subplot(122, projection="3d")
    graficar_palabras_3d(ax2, X_tfidf[:, posiciones], vocab_grafica,
                         "Espacio TF-IDF 3D (Importancia)", "teal")

    plt.tight_layout()
    plt.savefig(RUTA_GRAFICA, dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
