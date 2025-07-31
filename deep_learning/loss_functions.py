from __future__ import division
import numpy as np
from MLearning.utils import accuracy_score
from MLearning.deep_learning.activation_functions import Sigmoid

class Perdida(object):
    def perdida(self, y_real, y_pred):
        return NotImplementedError()

    def gradiente(self, y, y_pred):
        raise NotImplementedError()

    def precision(self, y, y_pred):
        return 0

class PerdidaCuadratica(Perdida):
    def __init__(self): pass

    def perdida(self, y, y_pred):
        return 0.5 * np.power((y - y_pred), 2)

    def gradiente(self, y, y_pred):
        return -(y - y_pred)

class EntropiaCruzada(Perdida):
    def __init__(self): pass

    def perdida(self, y, p):
        # Evitar división por cero
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - y * np.log(p) - (1 - y) * np.log(1 - p)

    def precision(self, y, p):
        return accuracy_score(np.argmax(y, axis=1), np.argmax(p, axis=1))

    def gradiente(self, y, p):
        # Evitar división por cero
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - (y / p) + (1 - y) / (1 - p)