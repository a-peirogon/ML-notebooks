from __future__ import division, print_function
import numpy as np
import progressbar

# Importar funciones auxiliares
from MLearning.utils import train_test_split, standardize, to_categorical
from MLearning.utils import mean_squared_error, accuracy_score
from MLearning.deep_learning.loss_functions import SquareLoss, CrossEntropy
from MLearning.supervised_learning.decision_tree import RegressionTree
from MLearning.utils.misc import bar_widgets


class GradientBoosting(object):
    """Clase base para GradientBoostingClassifier y GradientBoostingRegressor.
    Utiliza un conjunto de árboles de regresión que se entrenan para predecir
    el gradiente de la función de pérdida.

    Parámetros:
    -----------
    n_estimators: int
        Número de árboles de regresión a utilizar.
    learning_rate: float
        Tasa de aprendizaje para la actualización de los pesos.
    min_samples_split: int
        Mínimo número de muestras requeridas para dividir un nodo.
    min_impurity: float
        Mínima impureza requerida para dividir un nodo.
    max_depth: int
        Profundidad máxima de los árboles.
    regression: bool
        True para regresión, False para clasificación.
    """
    def __init__(self, n_estimators, learning_rate, min_samples_split,
                 min_impurity, max_depth, regression):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.min_samples_split = min_samples_split
        self.min_impurity = min_impurity
        self.max_depth = max_depth
        self.regression = regression
        self.bar = progressbar.ProgressBar(widgets=bar_widgets)
        
        # Función de pérdida
        self.loss = SquareLoss() if regression else CrossEntropy()

        # Inicializar árboles de regresión
        self.trees = []
        for _ in range(n_estimators):
            tree = RegressionTree(
                    min_samples_split=self.min_samples_split,
                    min_impurity=min_impurity,
                    max_depth=self.max_depth)
            self.trees.append(tree)

    def fit(self, X, y):
        """Entrena el modelo con los datos X e y"""
        y_pred = np.full(np.shape(y), np.mean(y, axis=0))
        for i in self.bar(range(self.n_estimators)):
            # Calcular gradiente de la función de pérdida
            gradient = self.loss.gradient(y, y_pred)
            # Entrenar árbol para predecir el gradiente
            self.trees[i].fit(X, gradient)
            # Actualizar predicciones
            update = self.trees[i].predict(X)
            y_pred -= np.multiply(self.learning_rate, update)

    def predict(self, X):
        """Realiza predicciones para las muestras en X"""
        y_pred = np.array([])
        for tree in self.trees:
            update = tree.predict(X)
            update = np.multiply(self.learning_rate, update)
            y_pred = -update if not y_pred.any() else y_pred - update

        if not self.regression:
            # Convertir a distribución de probabilidad (clasificación)
            y_pred = np.exp(y_pred) / np.expand_dims(np.sum(np.exp(y_pred), axis=1), axis=1)
            y_pred = np.argmax(y_pred, axis=1)
        return y_pred


class GradientBoostingRegressor(GradientBoosting):
    """Gradient Boosting para problemas de regresión"""
    def __init__(self, n_estimators=200, learning_rate=0.5, min_samples_split=2,
                 min_var_red=1e-7, max_depth=4, debug=False):
        super(GradientBoostingRegressor, self).__init__(
            n_estimators=n_estimators, 
            learning_rate=learning_rate,
            min_samples_split=min_samples_split,
            min_impurity=min_var_red,
            max_depth=max_depth,
            regression=True)


class GradientBoostingClassifier(GradientBoosting):
    """Gradient Boosting para problemas de clasificación"""
    def __init__(self, n_estimators=200, learning_rate=0.5, min_samples_split=2,
                 min_info_gain=1e-7, max_depth=2, debug=False):
        super(GradientBoostingClassifier, self).__init__(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            min_samples_split=min_samples_split,
            min_impurity=min_info_gain,
            max_depth=max_depth,
            regression=False)

    def fit(self, X, y):
        """Preprocesa etiquetas y entrena el modelo"""
        y = to_categorical(y)
        super(GradientBoostingClassifier, self).fit(X, y)