from __future__ import print_function, division
import numpy as np
from MLearning.utils import calculate_covariance_matrix, normalize, standardize

class LDA():
    """Clasificador de Análisis Discriminante Lineal (LDA), también conocido como 
    discriminante lineal de Fisher. Además de clasificación, puede usarse para 
    reducción de dimensionalidad.

    Atributos:
    ----------
    w: array
        Vector de proyección que maximiza la separación entre clases
    """
    def __init__(self):
        self.w = None

    def transformar(self, X, y):
        """Transforma los datos proyectándolos sobre el vector discriminante.
        
        Parámetros:
        -----------
        X: array-like
            Datos a transformar
        y: array-like
            Etiquetas de clase
            
        Devuelve:
        --------
        array: Datos proyectados
        """
        self.ajustar(X, y)
        return X.dot(self.w)

    def ajustar(self, X, y):
        """Calcula el vector discriminante óptimo.
        
        Parámetros:
        -----------
        X: array-like
            Datos de entrenamiento
        y: array-like
            Etiquetas de clase (binarias)
        """
        # Separar datos por clase
        X1 = X[y == 0]
        X2 = X[y == 1]

        # Calcular matrices de covarianza
        cov1 = calculate_covariance_matrix(X1)
        cov2 = calculate_covariance_matrix(X2)
        cov_total = cov1 + cov2

        # Calcular medias de clase
        media1 = X1.mean(axis=0)
        media2 = X2.mean(axis=0)
        diff_medias = np.atleast_1d(media1 - media2)

        # Calcular vector discriminante óptimo
        self.w = np.linalg.pinv(cov_total).dot(diff_medias)

    def predecir(self, X):
        """Realiza predicciones de clase para nuevos datos.
        
        Parámetros:
        -----------
        X: array-like
            Datos a clasificar
            
        Devuelve:
        --------
        array: Predicciones de clase (0 o 1)
        """
        # Calcular proyecciones
        h = X.dot(self.w)
        # Clasificar según el signo de la proyección
        return (h < 0).astype(int)
    
    def distancia_a_hiperplano(self, X):
        """Calcula la distancia de cada muestra al hiperplano discriminante.
        
        Parámetros:
        -----------
        X: array-like
            Datos de entrada
            
        Devuelve:
        --------
        array: Distancias al hiperplano
        """
        if self.w is None:
            raise ValueError("Primero debe ajustar el modelo (fit)")
        return X.dot(self.w) / np.linalg.norm(self.w)