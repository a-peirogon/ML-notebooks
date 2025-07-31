from __future__ import print_function, division
import math
import numpy as np

# Importar funciones auxiliares
from MLearning.utils import train_test_split, to_categorical, normalize, accuracy_score
from MLearning.deep_learning.activation_functions import Sigmoid, ReLU, SoftPlus, LeakyReLU, TanH, ELU
from MLearning.deep_learning.loss_functions import CrossEntropy, SquareLoss
from MLearning.utils import Plot
from MLearning.utils.misc import bar_widgets
import progressbar

class Perceptron():
    """Perceptrón: Clasificador de red neuronal de una capa.

    Parámetros:
    -----------
    n_iteraciones: int (default=20000)
        Número de iteraciones de entrenamiento.
    funcion_activacion: class (default=Sigmoid)
        Función de activación para las neuronas.
        Opciones: Sigmoid, ReLU, LeakyReLU, SoftPlus, TanH, ELU
    funcion_perdida: class (default=SquareLoss)
        Función de pérdida para evaluar el modelo.
        Opciones: SquareLoss, CrossEntropy
    tasa_aprendizaje: float (default=0.01)
        Tamaño del paso para actualizar los pesos.
    """
    
    def __init__(self, n_iteraciones=20000, funcion_activacion=Sigmoid, 
                 funcion_perdida=SquareLoss, tasa_aprendizaje=0.01):
        self.n_iteraciones = n_iteraciones
        self.tasa_aprendizaje = tasa_aprendizaje
        self.funcion_perdida = funcion_perdida()
        self.funcion_activacion = funcion_activacion()
        self.barra_progreso = progressbar.ProgressBar(widgets=bar_widgets)

    def ajustar(self, X, y):
        """Entrena el modelo con los datos proporcionados.
        
        Parámetros:
        -----------
        X: array-like
            Datos de entrenamiento
        y: array-like
            Etiquetas de clase (one-hot encoded)
        """
        n_muestras, n_caracteristicas = np.shape(X)
        _, n_salidas = np.shape(y)

        # Inicializar pesos entre [-1/sqrt(N), 1/sqrt(N)]
        limite = 1 / math.sqrt(n_caracteristicas)
        self.W = np.random.uniform(-limite, limite, (n_caracteristicas, n_salidas))
        self.w0 = np.zeros((1, n_salidas))

        for i in self.barra_progreso(range(self.n_iteraciones)):
            # Paso hacia adelante
            salida_lineal = X.dot(self.W) + self.w0
            y_pred = self.funcion_activacion(salida_lineal)
            
            # Paso hacia atrás
            gradiente_error = self.funcion_perdida.gradiente(y, y_pred) * \
                            self.funcion_activacion.gradiente(salida_lineal)
            
            # Calcular gradientes
            grad_W = X.T.dot(gradiente_error)
            grad_w0 = np.sum(gradiente_error, axis=0, keepdims=True)
            
            # Actualizar pesos
            self.W -= self.tasa_aprendizaje * grad_W
            self.w0 -= self.tasa_aprendizaje * grad_w0

    def predecir(self, X):
        """Realiza predicciones sobre nuevos datos.
        
        Parámetros:
        -----------
        X: array-like
            Datos a predecir
            
        Devuelve:
        --------
        array: Predicciones del modelo
        """
        return self.funcion_activacion(X.dot(self.W) + self.w0)