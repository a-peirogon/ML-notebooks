from __future__ import print_function, division
import numpy as np
import math
from sklearn import datasets

from MLearning.utils import train_test_split, to_categorical, normalize, accuracy_score, Plot
from MLearning.deep_learning.activation_functions import Sigmoid, Softmax
from MLearning.deep_learning.loss_functions import CrossEntropy

class PerceptronMulticapa():
    """Clasificador Perceptrón Multicapa. Red neuronal totalmente conectada con una capa oculta.
    
    Parámetros:
    -----------
    n_ocultas: int
        Número de neuronas en la capa oculta.
    n_iteraciones: int (default=3000)
        Número de iteraciones de entrenamiento.
    tasa_aprendizaje: float (default=0.01)
        Tasa de aprendizaje para la actualización de pesos.
    """
    def __init__(self, n_ocultas, n_iteraciones=3000, tasa_aprendizaje=0.01):
        self.n_ocultas = n_ocultas
        self.n_iteraciones = n_iteraciones
        self.tasa_aprendizaje = tasa_aprendizaje
        self.activacion_oculta = Sigmoid()
        self.activacion_salida = Softmax()
        self.funcion_perdida = CrossEntropy()

    def _inicializar_pesos(self, X, y):
        n_muestras, n_caracteristicas = X.shape
        _, n_salidas = y.shape
        
        # Capa oculta
        limite = 1 / math.sqrt(n_caracteristicas)
        self.W = np.random.uniform(-limite, limite, (n_caracteristicas, self.n_ocultas))
        self.w0 = np.zeros((1, self.n_ocultas))
        
        # Capa de salida
        limite = 1 / math.sqrt(self.n_ocultas)
        self.V = np.random.uniform(-limite, limite, (self.n_ocultas, n_salidas))
        self.v0 = np.zeros((1, n_salidas))

    def ajustar(self, X, y):
        """Entrena el modelo con los datos de entrada.
        
        Parámetros:
        -----------
        X: array-like
            Datos de entrenamiento
        y: array-like
            Etiquetas one-hot codificadas
        """
        self._inicializar_pesos(X, y)

        for i in range(self.n_iteraciones):
            # -------------------
            #  Paso hacia adelante
            # -------------------
            
            # Capa oculta
            entrada_oculta = X.dot(self.W) + self.w0
            salida_oculta = self.activacion_oculta(entrada_oculta)
            
            # Capa de salida
            entrada_salida = salida_oculta.dot(self.V) + self.v0
            y_pred = self.activacion_salida(entrada_salida)

            # -------------------
            #  Paso hacia atrás
            # -------------------
            
            # Capa de salida
            grad_entrada_salida = self.funcion_perdida.gradiente(y, y_pred) * self.activacion_salida.gradiente(entrada_salida)
            grad_V = salida_oculta.T.dot(grad_entrada_salida)
            grad_v0 = np.sum(grad_entrada_salida, axis=0, keepdims=True)
            
            # Capa oculta
            grad_entrada_oculta = grad_entrada_salida.dot(self.V.T) * self.activacion_oculta.gradiente(entrada_oculta)
            grad_W = X.T.dot(grad_entrada_oculta)
            grad_w0 = np.sum(grad_entrada_oculta, axis=0, keepdims=True)

            # Actualización de pesos
            self.V -= self.tasa_aprendizaje * grad_V
            self.v0 -= self.tasa_aprendizaje * grad_v0
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
        array: Probabilidades para cada clase
        """
        # Paso hacia adelante
        entrada_oculta = X.dot(self.W) + self.w0
        salida_oculta = self.activacion_oculta(entrada_oculta)
        entrada_salida = salida_oculta.dot(self.V) + self.v0
        return self.activacion_salida(entrada_salida)


def main():
    # Cargar y preparar datos
    data = datasets.load_digits()
    X = normalize(data.data)
    y = data.target
    y = to_categorical(y)

    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, seed=1)

    # Crear y entrenar modelo
    modelo = PerceptronMulticapa(n_ocultas=16,
                               n_iteraciones=1000,
                               tasa_aprendizaje=0.01)
    modelo.ajustar(X_train, y_train)

    # Evaluar modelo
    y_pred = np.argmax(modelo.predecir(X_test), axis=1)
    y_test = np.argmax(y_test, axis=1)
    precision = accuracy_score(y_test, y_pred)
    print("Precisión:", precision)

    # Visualizar resultados
    Plot().plot_in_2d(X_test, y_pred, 
                     title="Perceptrón Multicapa", 
                     accuracy=precision, 
                     legend_labels=np.unique(y))

if __name__ == "__main__":
    main()