import os

import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

RUTA_CORPUS = "corpus_lematizado.txt"
RUTA_MODELO = "modelo_word2vec.model"
PALABRAS_A_GRAFICAR = 60
PALABRAS_DE_CONSULTA = ["war", "general", "enemy", "army", "victory", "spy", "heaven", "attack"]


def cargar_oraciones(ruta):
    with open(ruta, encoding="utf-8") as archivo:
        lineas = archivo.read().split("\n")
    return [linea.split() for linea in lineas if linea.strip() != ""]


def entrenar_modelo(oraciones):
    modelo = Word2Vec(
        sentences=oraciones,
        vector_size=100,
        window=5,
        min_count=3,
        sg=1,
        epochs=50,
        workers=1,
        seed=42
    )
    return modelo


def explorar_modelo(modelo):
    print("Vocabulario del modelo:", len(modelo.wv))
    print("Dimensiones de cada vector:", modelo.wv.vector_size)

    for palabra in PALABRAS_DE_CONSULTA:
        if palabra in modelo.wv:
            print(f"\nPalabras mas parecidas a '{palabra}':")
            for vecina, similitud in modelo.wv.most_similar(palabra, topn=5):
                print(f"  {vecina}: {similitud:.3f}")

    if "general" in modelo.wv and "army" in modelo.wv:
        similitud = modelo.wv.similarity("general", "army")
        print(f"\nSimilitud coseno entre 'general' y 'army': {similitud:.3f}")


def graficar_2d(coordenadas, palabras, titulo, ruta, color):
    plt.figure(figsize=(12, 9))
    plt.scatter(coordenadas[:, 0], coordenadas[:, 1], c=color, s=70, edgecolors="k", alpha=0.8)

    for i, palabra in enumerate(palabras):
        plt.annotate(palabra, (coordenadas[i, 0], coordenadas[i, 1]),
                     xytext=(4, 4), textcoords="offset points", fontsize=9)

    plt.title(titulo)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(ruta, dpi=150)
    plt.close()


def graficar_3d(coordenadas, palabras, titulo, ruta, color):
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(coordenadas[:, 0], coordenadas[:, 1], coordenadas[:, 2],
               c=color, s=70, edgecolors="k", alpha=0.8, depthshade=True)

    for i, palabra in enumerate(palabras):
        ax.text(coordenadas[i, 0], coordenadas[i, 1], coordenadas[i, 2] + 0.05,
                palabra, fontsize=9)

    ax.set_title(titulo)
    ax.set_xlabel("Comp. Principal 1")
    ax.set_ylabel("Comp. Principal 2")
    ax.set_zlabel("Comp. Principal 3")
    plt.tight_layout()
    plt.savefig(ruta, dpi=150)
    plt.close()


def main():
    if not os.path.exists(RUTA_CORPUS):
        print("No existe", RUTA_CORPUS, "- ejecuta primero vectorizacion.py")
        return

    oraciones = cargar_oraciones(RUTA_CORPUS)
    print("Oraciones cargadas:", len(oraciones))

    modelo = entrenar_modelo(oraciones)
    modelo.save(RUTA_MODELO)
    explorar_modelo(modelo)

    palabras = modelo.wv.index_to_key[:PALABRAS_A_GRAFICAR]
    vectores = np.array([modelo.wv[palabra] for palabra in palabras])

    coords_pca_2d = PCA(n_components=2).fit_transform(vectores)
    graficar_2d(coords_pca_2d, palabras,
                "Word2Vec - PCA 2D (palabras mas frecuentes)",
                "word2vec_pca_2d.png", "teal")

    coords_pca_3d = PCA(n_components=3).fit_transform(vectores)
    graficar_3d(coords_pca_3d, palabras,
                "Word2Vec - PCA 3D (palabras mas frecuentes)",
                "word2vec_pca_3d.png", "purple")

    perplejidad = min(15, len(palabras) - 1)
    coords_tsne = TSNE(n_components=2, perplexity=perplejidad, random_state=42).fit_transform(vectores)
    graficar_2d(coords_tsne, palabras,
                "Word2Vec - t-SNE 2D (palabras mas frecuentes)",
                "word2vec_tsne_2d.png", "crimson")

    print("\nImagenes guardadas: word2vec_pca_2d.png, word2vec_pca_3d.png, word2vec_tsne_2d.png")


if __name__ == "__main__":
    main()
