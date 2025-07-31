from __future__ import division, print_function
import numpy as np
import math
from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd

# Importar funciones auxiliares
from MLearning.utils import train_test_split, accuracy_score, Plot

# Toque de decisión (Decision Stump) usado como clasificador débil en esta implementación de Adaboost
class DecisionStump():
    def __init__(self):
        # Determina si la muestra se clasificará como -1 o 1 dado un umbral
        self.polaridad = 1
        # Índice de la característica usada para la clasificación
        self.indice_caracteristica = None
        # Valor umbral contra el que se compara la característica
        self.umbral = None
        # Valor que indica la precisión del clasificador (peso)
        self.alfa = None
        
        class Adaboost():
    """Método de boosting que utiliza un conjunto de clasificadores débiles
    para crear un clasificador fuerte. Esta implementación usa toques de decisión
    (decision stumps), que son árboles de decisión de un solo nivel.

    Parámetros:
    -----------
    n_clf: int
        Número de clasificadores débiles que se utilizarán.
    """
    def __init__(self, n_clf=5):
        self.n_clf = n_clf

    def fit(self, X, y):
        n_samples, n_features = np.shape(X)

        # Inicializar pesos a 1/N
        w = np.full(n_samples, (1 / n_samples))
        
        self.clfs = []
        # Iterar a través de los clasificadores
        for _ in range(self.n_clf):
            clf = DecisionStump()
            # Error mínimo encontrado al usar un umbral determinado
            min_error = float('inf')
            
            # Probar cada valor único de característica como umbral
            for feature_i in range(n_features):
                feature_values = np.expand_dims(X[:, feature_i], axis=1)
                unique_values = np.unique(feature_values)
                
                for threshold in unique_values:
                    p = 1
                    prediction = np.ones(np.shape(y))
                    prediction[X[:, feature_i] < threshold] = -1
                    error = sum(w[y != prediction])
                    
                    # Si el error es mayor al 50%, invertir la polaridad
                    if error > 0.5:
                        error = 1 - error
                        p = -1

                    # Guardar configuración si encontramos menor error
                    if error < min_error:
                        clf.polarity = p
                        clf.threshold = threshold
                        clf.feature_index = feature_i
                        min_error = error
            
            # Calcular alpha (peso del clasificador)
            clf.alpha = 0.5 * math.log((1.0 - min_error) / (min_error + 1e-10))
            
            # Actualizar pesos de las muestras
            predictions = np.ones(np.shape(y))
            negative_idx = (clf.polarity * X[:, clf.feature_index] < clf.polarity * clf.threshold)
            predictions[negative_idx] = -1
            w *= np.exp(-clf.alpha * y * predictions)
            w /= np.sum(w)

            self.clfs.append(clf)

    def predict(self, X):
        n_samples = np.shape(X)[0]
        y_pred = np.zeros((n_samples, 1))
        
        for clf in self.clfs:
            predictions = np.ones(np.shape(y_pred))
            negative_idx = (clf.polarity * X[:, clf.feature_index] < clf.polarity * clf.threshold)
            predictions[negative_idx] = -1
            y_pred += clf.alpha * predictions

        return np.sign(y_pred).flatten()


def main():
    # Cargar dataset de dígitos
    data = datasets.load_digits()
    X = data.data
    y = data.target

    # Preparar datos para clasificación binaria (dígitos 1 vs 8)
    digit1 = 1
    digit2 = 8
    idx = np.append(np.where(y == digit1)[0], np.where(y == digit2)[0])
    y = data.target[idx]
    y[y == digit1] = -1  # Etiqueta -1 para dígito 1
    y[y == digit2] = 1   # Etiqueta 1 para dígito 8
    X = data.data[idx]

    # Dividir datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    # Entrenar Adaboost con 5 clasificadores débiles
    clf = Adaboost(n_clf=5)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Calcular y mostrar precisión
    accuracy = accuracy_score(y_test, y_pred)
    print("Precisión:", accuracy)

    # Visualizar resultados en 2D usando PCA
    Plot().plot_in_2d(X_test, y_pred, title="Adaboost", accuracy=accuracy)


if __name__ == "__main__":
    main()