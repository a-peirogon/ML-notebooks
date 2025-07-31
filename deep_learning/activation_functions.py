import numpy as np

# Colección de funciones de activación
# Referencia: https://es.wikipedia.org/wiki/Funci%C3%B3n_de_activaci%C3%B3n

class Sigmoide():
    """Función Sigmoide que mapea valores a un rango entre 0 y 1."""
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def gradiente(self, x):
        """Calcula el gradiente de la función sigmoide."""
        return self.__call__(x) * (1 - self.__call__(x))

class Softmax():
    """Función Softmax para normalizar valores a una distribución de probabilidad."""
    def __call__(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def gradiente(self, x):
        """Calcula el gradiente de la función softmax."""
        p = self.__call__(x)
        return p * (1 - p)

class TangenteHiperbolica():
    """Función Tangente Hiperbólica que mapea valores a un rango entre -1 y 1."""
    def __call__(self, x):
        return 2 / (1 + np.exp(-2*x)) - 1

    def gradiente(self, x):
        """Calcula el gradiente de la tangente hiperbólica."""
        return 1 - np.power(self.__call__(x), 2)

class ReLU():
    """Unidad Lineal Rectificada (ReLU) que devuelve el máximo entre 0 y el valor de entrada."""
    def __call__(self, x):
        return np.where(x >= 0, x, 0)

    def gradiente(self, x):
        """Calcula el gradiente de ReLU."""
        return np.where(x >= 0, 1, 0)

class LeakyReLU():
    """Variante de ReLU que permite un pequeño gradiente para valores negativos."""
    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def __call__(self, x):
        return np.where(x >= 0, x, self.alpha * x)

    def gradiente(self, x):
        """Calcula el gradiente de LeakyReLU."""
        return np.where(x >= 0, 1, self.alpha)

class ELU():
    """Unidad Exponencial Lineal que permite valores negativos con una saturación suave."""
    def __init__(self, alpha=0.1):
        self.alpha = alpha 

    def __call__(self, x):
        return np.where(x >= 0.0, x, self.alpha * (np.exp(x) - 1))

    def gradiente(self, x):
        """Calcula el gradiente de ELU."""
        return np.where(x >= 0.0, 1, self.__call__(x) + self.alpha)

class SELU():
    """Unidad Exponencial Lineal Escalada que permite auto-normalización de redes neuronales.
    Referencia: https://arxiv.org/abs/1706.02515"""
    def __init__(self):
        self.alpha = 1.6732632423543772848170429916717
        self.scale = 1.0507009873554804934193349852946 

    def __call__(self, x):
        return self.scale * np.where(x >= 0.0, x, self.alpha*(np.exp(x)-1))

    def gradiente(self, x):
        """Calcula el gradiente de SELU."""
        return self.scale * np.where(x >= 0.0, 1, self.alpha * np.exp(x))

class SoftPlus():
    """Función SoftPlus, una aproximación suave de la función ReLU."""
    def __call__(self, x):
        return np.log(1 + np.exp(x))

    def gradiente(self, x):
        """Calcula el gradiente de SoftPlus (equivalente a la función sigmoide)."""
        return 1 / (1 + np.exp(-x))