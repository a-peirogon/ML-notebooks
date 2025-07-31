from __future__ import print_function, division
from terminaltables import AsciiTable
import numpy as np
import progressbar
from MLearning.utils import batch_iterator
from MLearning.utils.misc import bar_widgets


class RedNeuronal():
    """Red Neuronal. Modelo base de Aprendizaje Profundo.

    Parámetros:
    -----------
    optimizador: clase
        Optimizador de pesos que se usará para ajustar los pesos con el fin de minimizar
        la pérdida.
    perdida: clase
        Función de pérdida usada para medir el desempeño del modelo. PerdidaCuadratica o EntropiaCruzada.
    validacion: tuple
        Tupla que contiene datos y etiquetas de validación (X, y)
    """
    def __init__(self, optimizador, perdida, datos_validacion=None):
        self.optimizador = optimizador
        self.capas = []
        self.errores = {"entrenamiento": [], "validacion": []}
        self.funcion_perdida = perdida()
        self.barra_progreso = progressbar.ProgressBar(widgets=bar_widgets)

        self.conjunto_val = None
        if datos_validacion:
            X, y = datos_validacion
            self.conjunto_val = {"X": X, "y": y}

    def establecer_entrenable(self, entrenable):
        """ Método que permite congelar los pesos de las capas de la red. """
        for capa in self.capas:
            capa.entrenable = entrenable

    def agregar(self, capa):
        """ Método que añade una capa a la red neuronal """
        # Si no es la primera capa, establecer la forma de entrada
        # como la forma de salida de la última capa añadida
        if self.capas:
            capa.establecer_forma_entrada(forma=self.capas[-1].forma_salida())

        # Si la capa tiene pesos que necesitan inicialización
        if hasattr(capa, 'inicializar'):
            capa.inicializar(optimizador=self.optimizador)

        # Añadir capa a la red
        self.capas.append(capa)

    def evaluar_lote(self, X, y):
        """ Evalúa el modelo sobre un único lote de muestras """
        y_pred = self._propagacion_adelante(X, entrenamiento=False)
        perdida = np.mean(self.funcion_perdida.perdida(y, y_pred))
        precision = self.funcion_perdida.precision(y, y_pred)

        return perdida, precision

    def entrenar_lote(self, X, y):
        """ Actualización única del gradiente sobre un lote de muestras """
        y_pred = self._propagacion_adelante(X)
        perdida = np.mean(self.funcion_perdida.perdida(y, y_pred))
        precision = self.funcion_perdida.precision(y, y_pred)
        # Calcular el gradiente de la función de pérdida respecto a y_pred
        grad_perdida = self.funcion_perdida.gradiente(y, y_pred)
        # Retropropagación. Actualizar pesos
        self._propagacion_atras(grad_perdida=grad_perdida)

        return perdida, precision

    def entrenar(self, X, y, n_epocas, tamano_lote):
        """ Entrena el modelo por un número fijo de épocas """
        for _ in self.barra_progreso(range(n_epocas)):
            
            error_lote = []
            for X_lote, y_lote in batch_iterator(X, y, batch_size=tamano_lote):
                perdida, _ = self.entrenar_lote(X_lote, y_lote)
                error_lote.append(perdida)

            self.errores["entrenamiento"].append(np.mean(error_lote))

            if self.conjunto_val is not None:
                perdida_val, _ = self.evaluar_lote(self.conjunto_val["X"], self.conjunto_val["y"])
                self.errores["validacion"].append(perdida_val)

        return self.errores["entrenamiento"], self.errores["validacion"]

    def _propagacion_adelante(self, X, entrenamiento=True):
        """ Calcula la salida de la red neuronal """
        salida_capa = X
        for capa in self.capas:
            salida_capa = capa.propagacion_adelante(salida_capa, entrenamiento)

        return salida_capa

    def _propagacion_atras(self, grad_perdida):
        """ Propaga el gradiente 'hacia atrás' y actualiza los pesos en cada capa """
        for capa in reversed(self.capas):
            grad_perdida = capa.propagacion_atras(grad_perdida)

    def resumen(self, nombre="Resumen del Modelo"):
        # Imprimir nombre del modelo
        print(AsciiTable([[nombre]]).table)
        # Forma de entrada de la red (forma de entrada de la primera capa)
        print("Forma de Entrada: %s" % str(self.capas[0].forma_entrada))
        # Iterar a través de la red y obtener configuración de cada capa
        datos_tabla = [["Tipo de Capa", "Parámetros", "Forma de Salida"]]
        total_param = 0
        for capa in self.capas:
            nombre_capa = capa.nombre_capa()
            parametros = capa.parametros()
            forma_salida = capa.forma_salida()
            datos_tabla.append([nombre_capa, str(parametros), str(forma_salida)])
            total_param += parametros
        # Imprimir tabla de configuración de la red
        print(AsciiTable(datos_tabla).table)
        print("Parámetros Totales: %d\n" % total_param)

    def predecir(self, X):
        """ Usa el modelo entrenado para predecir etiquetas de X """
        return self._propagacion_adelante(X, entrenamiento=False)