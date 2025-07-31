from __future__ import print_function, division
import numpy as np
from MLearning.utils import euclidean_distance

class KNN():
    """Clasificador K Vecinos Más Cercanos (K-Nearest Neighbors).

    Parámetros:
    -----------
    k: int
        Número de vecinos más cercanos que determinarán la clase
        de la muestra que queremos predecir.
    """
    def __init__(self, k=5):
        self.k = k

    def _votacion(self, etiquetas_vecinos):
        """Devuelve la clase más común entre las muestras vecinas"""
        conteos = np.bincount(etiquetas_vecinos.astype('int'))
        return conteos.argmax()

    def predecir(self, X_prueba, X_entrenamiento, y_entrenamiento):
        """Realiza predicciones para muestras de prueba.

        Parámetros:
        -----------
        X_prueba: array
            Conjunto de datos de prueba
        X_entrenamiento: array
            Conjunto de datos de entrenamiento
        y_entrenamiento: array
            Etiquetas del conjunto de entrenamiento

        Devuelve:
        --------
        array: Predicciones para las muestras de prueba
        """
        y_prediccion = np.empty(X_prueba.shape[0])
        
        # Determinar la clase de cada muestra
        for i, muestra in enumerate(X_prueba):
            # Ordenar muestras por distancia y obtener las K más cercanas
            indices = np.argsort([euclidean_distance(muestra, x) for x in X_entrenamiento])[:self.k]
            # Obtener etiquetas de los K vecinos más cercanos
            vecinos_cercanos = np.array([y_entrenamiento[i] for i in indices])
            # Asignar la clase más frecuente
            y_prediccion[i] = self._votacion(vecinos_cercanos)

        return y_prediccion