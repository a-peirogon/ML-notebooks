from __future__ import division, print_function
import numpy as np
import progressbar

from MLearning.utils import train_test_split, standardize, to_categorical, normalize
from MLearning.utils import mean_squared_error, accuracy_score
from MLearning.supervised_learning import XGBoostRegressionTree
from MLearning.deep_learning.activation_functions import Sigmoid
from MLearning.utils.misc import bar_widgets
from MLearning.utils import Plot


class PerdidaLogistica():
    """Función de pérdida logística para XGBoost en problemas de clasificación."""
    
    def __init__(self):
        sigmoide = Sigmoid()
        self.funcion_log = sigmoide
        self.gradiente_log = sigmoide.gradient

    def perdida(self, y, y_pred):
        """Calcula la pérdida logística."""
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)  # Evita valores extremos
        p = self.funcion_log(y_pred)
        return y * np.log(p) + (1 - y) * np.log(1 - p)

    def gradiente(self, y, y_pred):
        """Calcula el gradiente de la pérdida respecto a y_pred."""
        p = self.funcion_log(y_pred)
        return -(y - p)

    def hessiano(self, y, y_pred):
        """Calcula la matriz hessiana (derivadas segundas)."""
        p = self.funcion_log(y_pred)
        return p * (1 - p)


class XGBoost():
    """Implementación del algoritmo XGBoost para clasificación.

    Referencia: http://xgboost.readthedocs.io/en/latest/model.html

    Parámetros:
    -----------
    n_estimadores: int
        Número de árboles en el modelo.
    tasa_aprendizaje: float
        Tamaño del paso en la optimización por gradiente.
    min_muestras_div: int
        Mínimo número de muestras para dividir un nodo.
    min_impureza: float
        Mínima reducción de impureza para dividir un nodo.
    max_profundidad: int
        Profundidad máxima de los árboles.
    """
    
    def __init__(self, n_estimadores=200, tasa_aprendizaje=0.001, min_muestras_div=2,
                 min_impureza=1e-7, max_profundidad=2):
        self.n_estimadores = n_estimadores
        self.tasa_aprendizaje = tasa_aprendizaje
        self.min_muestras_div = min_muestras_div
        self.min_impureza = min_impureza
        self.max_profundidad = max_profundidad

        self.barra = progressbar.ProgressBar(widgets=bar_widgets)
        
        # Función de pérdida para clasificación
        self.perdida = PerdidaLogistica()

        # Inicializar árboles de regresión
        self.arboles = []
        for _ in range(n_estimadores):
            arbol = XGBoostRegressionTree(
                    min_samples_split=self.min_muestras_div,
                    min_impurity=self.min_impureza,
                    max_depth=self.max_profundidad,
                    loss=self.perdida)
            self.arboles.append(arbol)

    def ajustar(self, X, y):
        """Entrena el modelo XGBoost con los datos proporcionados.
        
        Args:
            X: Matriz de características de entrenamiento
            y: Vector de etiquetas
        """
        y = to_categorical(y)  # Convertir a one-hot encoding

        y_pred = np.zeros_like(y)
        for i in self.bar(range(self.n_estimadores)):
            arbol = self.arboles[i]
            # Concatenar etiquetas reales y predicciones
            y_con_pred = np.concatenate((y, y_pred), axis=1)
            arbol.fit(X, y_con_pred)
            actualizacion_pred = arbol.predict(X)
            
            # Actualizar predicciones
            y_pred -= np.multiply(self.tasa_aprendizaje, actualizacion_pred)

    def predecir(self, X):
        """Realiza predicciones sobre nuevos datos.
        
        Args:
            X: Matriz de características a predecir
            
        Returns:
            array: Predicciones de clase
        """
        y_pred = None
        for arbol in self.arboles:
            actualizacion_pred = arbol.predict(X)
            if y_pred is None:
                y_pred = np.zeros_like(actualizacion_pred)
            y_pred -= np.multiply(self.tasa_aprendizaje, actualizacion_pred)

        # Convertir a distribución de probabilidad (Softmax)
        y_pred = np.exp(y_pred) / np.sum(np.exp(y_pred), axis=1, keepdims=True)
        # Seleccionar la clase con mayor probabilidad
        return np.argmax(y_pred, axis=1)