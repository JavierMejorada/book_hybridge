import limpieza
import vectorizacion
import semantica_distribucional

print("=== 1. Carga y limpieza ===")
limpieza.main()

print("\n=== 2. Bag-of-Words y TF-IDF ===")
vectorizacion.main()

print("\n=== 3. Semantica distribucional (Word2Vec) ===")
semantica_distribucional.main()
