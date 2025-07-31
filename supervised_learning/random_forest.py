from __future__ import division, print_function
import numpy as np
import math
import progressbar

# Importar funciones auxiliares
from MLearning.utils import divide_on_feature, train_test_split, get_random_subsets, normalize
from MLearning.utils import accuracy_score, calculate_entropy
from MLearning.unsupervised_learning import PCA
from MLearning.supervised_learning import ClassificationTree
from MLearning.utils.misc import bar_widgets
from MLearning.utils import Plot


class BosqueAleatorio():
    """Clasificador Bosque Aleatorio. Utiliza un conjunto de árboles de decisión que
    se entrenan con subconjuntos aleatorios de los datos y características.

    Parámetros:
    -----------
    n_estimadores: int (default=100)
        Número de árboles de decisión en el bosque.
    max_caracteristicas: int (default=None)
        Máximo número de características por árbol. Si es None, usa sqrt(n_features).
    min_muestras_div: int (default=2)
        Mínimo número de muestras requeridas para dividir un nodo.
    min_ganancia: float (default=0)
        Mínima ganancia de información requerida para dividir un nodo.
    max_profundidad: int (default=inf)
        Profundidad máxima permitida para los árboles.
    """
    
    def __init__(self, n_estimadores=100, max_caracteristicas=None, min_muestras_div=2,
                 min_ganancia=0, max_profundidad=float("inf")):
        self.n_estimadores = n_estimadores
        self.max_caracteristicas = max_caracteristicas
        self.min_muestras_div = min_muestras_div
        self.min_ganancia = min_ganancia
        self.max_profundidad = max_profundidad
        self.barra_progreso = progressbar.ProgressBar(widgets=bar_widgets)

        # Inicializar árboles de decisión
        self.arboles = []
        for _ in range(n_estimadores):
            self.arboles.append(
                ClassificationTree(
                    min_samples_split=self.min_muestras_div,
                    min_impurity=self.min_ganancia,
                    max_depth=self.max_profundidad))

    def ajustar(self, X, y):
        """Entrena el bosque con los datos proporcionados.
        
        Parámetros:
        -----------
        X: array-like
            Datos de entrenamiento
        y: array-like
            Etiquetas de clase
        """
        n_caracteristicas = np.shape(X)[1]
        
        # Si max_caracteristicas no está definido, usar sqrt(n_caracteristicas)
        if not self.max_caracteristicas:
            self.max_caracteristicas = int(math.sqrt(n_caracteristicas))

        # Crear subconjuntos aleatorios de datos para cada árbol
        subconjuntos = get_random_subsets(X, y, self.n_estimadores)

        for i in self.barra_progreso(range(self.n_estimadores)):
            X_subconjunto, y_subconjunto = subconjuntos[i]
            
            # Selección aleatoria de características (feature bagging)
            idx = np.random.choice(range(n_caracteristicas), 
                                 size=self.max_caracteristicas, 
                                 replace=True)
            
            # Guardar índices de características para predicción
            self.arboles[i].feature_indices = idx
            
            # Seleccionar las características correspondientes
            X_subconjunto = X_subconjunto[:, idx]
            
            # Entrenar el árbol con el subconjunto
            self.arboles[i].fit(X_subconjunto, y_subconjunto)

    def predecir(self, X):
        """Realiza predicciones sobre nuevos datos.
        
        Parámetros:
        -----------
        X: array-like
            Datos a predecir
            
        Devuelve:
        --------
        array: Predicciones de clase
        """
        predicciones = np.empty((X.shape[0], len(self.arboles)))
        
        # Obtener predicciones de cada árbol
        for i, arbol in enumerate(self.arboles):
            idx = arbol.feature_indices
            prediccion = arbol.predict(X[:, idx])
            predicciones[:, i] = prediccion
            
        # Votación mayoritaria para la predicción final
        y_pred = []
        for predicciones_muestra in predicciones:
            y_pred.append(np.bincount(predicciones_muestra.astype('int')).argmax())
            
        return np.array(y_pred)