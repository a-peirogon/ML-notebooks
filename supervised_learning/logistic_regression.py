from __future__ import print_function, division
import numpy as np
import math
from MLearning.utils import make_diagonal, Plot
from MLearning.deep_learning.activation_functions import Sigmoid


class RegresionLogistica():
    """Clasificador de Regresión Logística.

    Parámetros:
    -----------
    tasa_aprendizaje: float (default=0.1)
        Tamaño del paso en el descenso del gradiente.
    descenso_gradiente: bool (default=True)
        Si es True, usa descenso de gradiente. Si es False, usa optimización por mínimos cuadrados.
    """
    def __init__(self, tasa_aprendizaje=0.1, descenso_gradiente=True):
        self.parametros = None
        self.tasa_aprendizaje = tasa_aprendizaje
        self.descenso_gradiente = descenso_gradiente
        self.sigmoide = Sigmoid()

    def _inicializar_parametros(self, X):
        """Inicializa los parámetros del modelo."""
        n_caracteristicas = np.shape(X)[1]
        # Inicializar parámetros entre [-1/sqrt(N), 1/sqrt(N)]
        limite = 1 / math.sqrt(n_caracteristicas)
        self.parametros = np.random.uniform(-limite, limite, (n_caracteristicas,))

    def ajustar(self, X, y, n_iteraciones=4000):
        """Entrena el modelo con los datos proporcionados.
        
        Parámetros:
        -----------
        X: array-like
            Datos de entrenamiento
        y: array-like
            Etiquetas de clase (0 o 1)
        n_iteraciones: int (default=4000)
            Número de iteraciones de entrenamiento
        """
        self._inicializar_parametros(X)
        
        for i in range(n_iteraciones):
            # Calcular predicción
            y_pred = self.sigmoide(X.dot(self.parametros))
            
            if self.descenso_gradiente:
                # Actualización por descenso de gradiente
                gradiente = X.T.dot(y - y_pred)
                self.parametros += self.tasa_aprendizaje * gradiente
            else:
                # Optimización por mínimos cuadrados
                diag_gradiente = make_diagonal(self.sigmoide.gradiente(X.dot(self.parametros)))
                self.parametros = np.linalg.pinv(X.T.dot(diag_gradiente).dot(X)).dot(X.T).dot(
                    diag_gradiente.dot(X).dot(self.parametros) + y - y_pred)

    def predecir(self, X):
        """Realiza predicciones sobre nuevos datos.
        
        Parámetros:
        -----------
        X: array-like
            Datos a predecir
            
        Devuelve:
        --------
        array: Predicciones de clase (0 o 1)
        """
        probabilidades = self.sigmoide(X.dot(self.parametros))
        return np.round(probabilidades).astype(int)

    def predecir_probabilidad(self, X):
        """Devuelve las probabilidades estimadas para cada clase.
        
        Parámetros:
        -----------
        X: array-like
            Datos a evaluar
            
        Devuelve:
        --------
        array: Probabilidades de pertenecer a la clase 1
        """
        return self.sigmoide(X.dot(self.parametros))