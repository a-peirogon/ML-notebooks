from __future__ import division, print_function
import numpy as np

from MLearning.utils import divide_on_feature, train_test_split, standardize, mean_squared_error
from MLearning.utils import calculate_entropy, accuracy_score, calculate_variance

class NodoDecision():
    """Clase que representa un nodo de decisión o hoja en el árbol de decisión

    Parámetros:
    -----------
    feature_i: int
        Índice de la característica que se usará como umbral.
    threshold: float
        Valor contra el que se compararán los valores de la característica.
    value: float
        Predicción de clase si es árbol de clasificación, o valor numérico si es regresión.
    true_branch: NodoDecision
        Siguiente nodo para muestras que cumplen el umbral.
    false_branch: NodoDecision
        Siguiente nodo para muestras que no cumplen el umbral.
    """
    def __init__(self, feature_i=None, threshold=None,
                 value=None, true_branch=None, false_branch=None):
        self.feature_i = feature_i          # Índice de la característica evaluada
        self.threshold = threshold          # Valor umbral para la característica
        self.value = value                  # Valor si el nodo es una hoja
        self.true_branch = true_branch      # Subárbol "izquierdo" (cumple condición)
        self.false_branch = false_branch    # Subárbol "derecho" (no cumple condición)


# Clase base para ArbolRegresion y ArbolClasificacion
class ArbolDecision(object):
    """Clase base para árboles de regresión y clasificación.

    Parámetros:
    -----------
    min_samples_split: int
        Mínimo número de muestras requeridas para dividir un nodo.
    min_impurity: float
        Mínima impureza requerida para dividir un nodo.
    max_depth: int
        Profundidad máxima del árbol.
    loss: function
        Función de pérdida usada en modelos de Gradient Boosting para calcular impureza.
    """
    def __init__(self, min_samples_split=2, min_impurity=1e-7,
                 max_depth=float("inf"), loss=None):
        self.raiz = None  # Nodo raíz del árbol
        self.min_samples_split = min_samples_split  # Mínimo de muestras para dividir
        self.min_impurity = min_impurity  # Mínima impureza para dividir
        self.max_depth = max_depth  # Profundidad máxima del árbol
        self._calculo_impureza = None  # Función para calcular impureza
        self._calculo_valor_hoja = None  # Función para determinar valor en hoja
        self.one_dim = None  # Si y es unidimensional
        self.loss = loss  # Función de pérdida para Gradient Boosting

    def ajustar(self, X, y, loss=None):
        """ Construye el árbol de decisión """
        self.one_dim = len(np.shape(y)) == 1
        self.raiz = self._construir_arbol(X, y)
        self.loss = None

    def _construir_arbol(self, X, y, profundidad_actual=0):
        """Método recursivo que construye el árbol dividiendo X e y según
        la característica que mejor separa los datos (basado en impureza)"""

        maxima_impureza = 0
        mejor_criterio = None  # Índice y umbral de característica
        mejores_conjuntos = None  # Subconjuntos de datos

        if len(np.shape(y)) == 1:
            y = np.expand_dims(y, axis=1)

        Xy = np.concatenate((X, y), axis=1)
        n_muestras, n_caracteristicas = np.shape(X)

        if n_muestras >= self.min_samples_split and profundidad_actual <= self.max_depth:
            for feature_i in range(n_caracteristicas):
                valores_caracteristica = np.expand_dims(X[:, feature_i], axis=1)
                valores_unicos = np.unique(valores_caracteristica)

                for threshold in valores_unicos:
                    Xy1, Xy2 = divide_on_feature(Xy, feature_i, threshold)

                    if len(Xy1) > 0 and len(Xy2) > 0:
                        y1 = Xy1[:, n_caracteristicas:]
                        y2 = Xy2[:, n_caracteristicas:]
                        impureza = self._calculo_impureza(y, y1, y2)

                        if impureza > maxima_impureza:
                            maxima_impureza = impureza
                            mejor_criterio = {"feature_i": feature_i, "threshold": threshold}
                            mejores_conjuntos = {
                                "leftX": Xy1[:, :n_caracteristicas],
                                "lefty": Xy1[:, n_caracteristicas:],
                                "rightX": Xy2[:, :n_caracteristicas],
                                "righty": Xy2[:, n_caracteristicas:]
                                }

        if maxima_impureza > self.min_impurity:
            rama_verdadera = self._construir_arbol(mejores_conjuntos["leftX"], 
                                                  mejores_conjuntos["lefty"], 
                                                  profundidad_actual + 1)
            rama_falsa = self._construir_arbol(mejores_conjuntos["rightX"], 
                                             mejores_conjuntos["righty"], 
                                             profundidad_actual + 1)
            return NodoDecision(feature_i=mejor_criterio["feature_i"], 
                              threshold=mejor_criterio["threshold"],
                              true_branch=rama_verdadera, 
                              false_branch=rama_falsa)

        valor_hoja = self._calculo_valor_hoja(y)
        return NodoDecision(value=valor_hoja)

    def predecir_valor(self, x, arbol=None):
        """Realiza una búsqueda recursiva por el árbol y devuelve
        el valor de la hoja donde termina la muestra"""

        if arbol is None:
            arbol = self.raiz

        if arbol.valor is not None:
            return arbol.valor

        valor_caracteristica = x[arbol.feature_i]
        rama = arbol.false_branch
        
        if isinstance(valor_caracteristica, (int, float)):
            if valor_caracteristica >= arbol.threshold:
                rama = arbol.true_branch
        elif valor_caracteristica == arbol.threshold:
            rama = arbol.true_branch

        return self.predecir_valor(x, rama)

    def predecir(self, X):
        """Predice las etiquetas para cada muestra en X"""
        return [self.predecir_valor(muestra) for muestra in X]

    def imprimir_arbol(self, arbol=None, indentacion=" "):
        """Imprime recursivamente la estructura del árbol"""
        if not arbol:
            arbol = self.raiz

        if arbol.valor is not None:
            print(arbol.valor)
        else:
            print("%s:%s? " % (arbol.feature_i, arbol.threshold))
            print("%sT->" % (indentacion), end="")
            self.imprimir_arbol(arbol.true_branch, indentacion + indentacion)
            print("%sF->" % (indentacion), end="")
            self.imprimir_arbol(arbol.false_branch, indentacion + indentacion)


class ArbolXGBoostRegresion(ArbolDecision):
    """Árbol de regresión para XGBoost"""

    def _dividir(self, y):
        """Divide y en valores reales (izquierda) y predicciones (derecha)"""
        col = int(np.shape(y)[1]/2)
        return y[:, :col], y[:, col:]

    def _ganancia(self, y, y_pred):
        numerador = np.power((y * self.loss.gradiente(y, y_pred)).sum(), 2)
        denominador = self.loss.hessiano(y, y_pred).sum()
        return 0.5 * (numerador / denominador)

    def _ganancia_por_taylor(self, y, y1, y2):
        y, y_pred = self._dividir(y)
        y1, y1_pred = self._dividir(y1)
        y2, y2_pred = self._dividir(y2)

        ganancia_verdadera = self._ganancia(y1, y1_pred)
        ganancia_falsa = self._ganancia(y2, y2_pred)
        ganancia_total = self._ganancia(y, y_pred)
        return ganancia_verdadera + ganancia_falsa - ganancia_total

    def _aproximacion_actualizacion(self, y):
        y, y_pred = self._dividir(y)
        gradiente = np.sum(y * self.loss.gradiente(y, y_pred), axis=0)
        hessiano = np.sum(self.loss.hessiano(y, y_pred), axis=0)
        return gradiente / hessiano

    def ajustar(self, X, y):
        self._calculo_impureza = self._ganancia_por_taylor
        self._calculo_valor_hoja = self._aproximacion_actualizacion
        super(ArbolXGBoostRegresion, self).ajustar(X, y)


class ArbolRegresion(ArbolDecision):
    def _calculo_reduccion_varianza(self, y, y1, y2):
        var_total = calculate_variance(y)
        var_1 = calculate_variance(y1)
        var_2 = calculate_variance(y2)
        fraccion_1 = len(y1) / len(y)
        fraccion_2 = len(y2) / len(y)
        return var_total - (fraccion_1 * var_1 + fraccion_2 * var_2)

    def _media_de_y(self, y):
        valor = np.mean(y, axis=0)
        return valor if len(valor) > 1 else valor[0]

    def ajustar(self, X, y):
        self._calculo_impureza = self._calculo_reduccion_varianza
        self._calculo_valor_hoja = self._media_de_y
        super(ArbolRegresion, self).ajustar(X, y)


class ArbolClasificacion(ArbolDecision):
    def _calculo_ganancia_informacion(self, y, y1, y2):
        p = len(y1) / len(y)
        entropia = calculate_entropy(y)
        return entropia - p * calculate_entropy(y1) - (1 - p) * calculate_entropy(y2)

    def _voto_mayoritario(self, y):
        mas_comun = None
        max_conteo = 0
        for etiqueta in np.unique(y):
            conteo = len(y[y == etiqueta])
            if conteo > max_conteo:
                mas_comun = etiqueta
                max_conteo = conteo
        return mas_comun

    def ajustar(self, X, y):
        self._calculo_impureza = self._calculo_ganancia_informacion
        self._calculo_valor_hoja = self._voto_mayoritario
        super(ArbolClasificacion, self).ajustar(X, y)