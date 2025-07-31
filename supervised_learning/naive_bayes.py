from __future__ import division, print_function
import numpy as np
import math
from MLearning.utils import train_test_split, normalize
from MLearning.utils import Plot, accuracy_score

class NaiveBayesGaussiano():
    """Clasificador Naive Bayes Gaussiano.
    
    Implementa el algoritmo de clasificación Naive Bayes asumiendo
    distribución normal para las características continuas.
    """
    
    def ajustar(self, X, y):
        """Ajusta el modelo a los datos de entrenamiento.
        
        Parámetros:
        -----------
        X: array-like
            Matriz de características de entrenamiento
        y: array-like
            Vector de etiquetas de clase
        """
        self.X, self.y = X, y
        self.clases = np.unique(y)
        self.parametros = []
        
        # Calcular media y varianza de cada característica por clase
        for i, c in enumerate(self.clases):
            # Seleccionar solo las muestras de la clase actual
            X_clase_c = X[np.where(y == c)]
            self.parametros.append([])
            
            # Calcular media y varianza para cada característica
            for columna in X_clase_c.T:
                parametros = {
                    "media": columna.mean(), 
                    "varianza": columna.var()
                }
                self.parametros[i].append(parametros)

    def _calcular_verosimilitud(self, media, varianza, x):
        """Calcula la verosimilitud gaussiana del dato x.
        
        Args:
            media: Media de la distribución
            varianza: Varianza de la distribución
            x: Valor de la característica
            
        Returns:
            float: Verosimilitud del dato bajo la distribución
        """
        eps = 1e-4  # Evita división por cero
        coeficiente = 1.0 / math.sqrt(2.0 * math.pi * varianza + eps)
        exponente = math.exp(-(math.pow(x - media, 2) / (2 * varianza + eps)))
        return coeficiente * exponente

    def _calcular_prior(self, clase):
        """Calcula la probabilidad a priori de una clase.
        
        Args:
            clase: Etiqueta de la clase
            
        Returns:
            float: Probabilidad a priori de la clase
        """
        return np.mean(self.y == clase)

    def _clasificar_muestra(self, muestra):
        """Clasifica una muestra usando el teorema de Bayes.
        
        P(Y|X) = P(X|Y)*P(Y) / P(X)
        
        Donde:
        - P(Y|X): Probabilidad posterior
        - P(X|Y): Verosimilitud (calculada como producto de verosimilitudes)
        - P(Y): Probabilidad a priori
        - P(X): Factor de escala (se omite)
        
        Args:
            muestra: Vector de características a clasificar
            
        Returns:
            int: Clase predicha
        """
        posteriores = []
        
        for i, clase in enumerate(self.clases):
            # Inicializar posterior con el prior
            posterior = self._calcular_prior(clase)
            
            # Calcular producto de verosimilitudes (supuesto naive)
            for valor, params in zip(muestra, self.parametros[i]):
                verosimilitud = self._calcular_verosimilitud(
                    params["media"], 
                    params["varianza"], 
                    valor
                )
                posterior *= verosimilitud
                
            posteriores.append(posterior)
            
        return self.clases[np.argmax(posteriores)]

    def predecir(self, X):
        """Realiza predicciones para un conjunto de muestras.
        
        Args:
            X: array-like
                Matriz de características a predecir
                
        Returns:
            array: Vector de predicciones
        """
        return [self._clasificar_muestra(muestra) for muestra in X]