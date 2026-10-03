import urllib.request

URL = "https://www.gutenberg.org/cache/epub/132/pg132.txt"
DESTINO = "arte_de_la_guerra.txt"

urllib.request.urlretrieve(URL, DESTINO)
print("Libro guardado en", DESTINO)
