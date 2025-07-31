from __future__ import print_function, division
import numpy as np
import math
from MLearning.utils import normalize, polynomial_features

class RegularizacionL1():
    """Regularización L1 para Regresión Lasso"""
    def __init__(self, alpha):
        self.alpha = alpha
    
    def __call__(self, w):
        return self.alpha * np.linalg.norm(w, ord=1)

    def gradiente(self, w):
        return self.alpha * np.sign(w)

class RegularizacionL2():
    """Regularización L2 para Regresión Ridge"""
    def __init__(self, alpha):
        self.alpha = alpha
    
    def __call__(self, w):
        return self.alpha * 0.5 * w.T.dot(w)

    def gradiente(self, w):
        return self.alpha * w

class RegularizacionElasticNet():
    """Regularización para Elastic Net que combina L1 y L2"""
    def __init__(self, alpha, ratio_l1=0.5):
        self.alpha = alpha
        self.ratio_l1 = ratio_l1

    def __call__(self, w):
        contrib_l1 = self.ratio_l1 * np.linalg.norm(w, ord=1)
        contrib_l2 = (1 - self.ratio_l1) * 0.5 * w.T.dot(w)
        return self.alpha * (contrib_l1 + contrib_l2)

    def gradiente(self, w):
        contrib_l1 = self.ratio_l1 * np.sign(w)
        contrib_l2 = (1 - self.ratio_l1) * w
        return self.alpha * (contrib_l1 + contrib_l2)

class RegresionBase(object):
    """Modelo base de regresión lineal"""
    def __init__(self, n_iteraciones, tasa_aprendizaje):
        self.n_iteraciones = n_iteraciones
        self.tasa_aprendizaje = tasa_aprendizaje

    def inicializar_pesos(self, n_caracteristicas):
        """Inicializa pesos aleatoriamente en el rango [-1/N, 1/N]"""
        limite = 1 / math.sqrt(n_caracteristicas)
        self.w = np.random.uniform(-limite, limite, (n_caracteristicas,))

    def ajustar(self, X, y):
        X = np.insert(X, 0, 1, axis=1)  # Añadir término de sesgo
        self.errores_entrenamiento = []
        self.inicializar_pesos(n_caracteristicas=X.shape[1])

        for i in range(self.n_iteraciones):
            y_pred = X.dot(self.w)
            mse = np.mean(0.5 * (y - y_pred)**2 + self.regularizacion(self.w))
            self.errores_entrenamiento.append(mse)
            
            gradiente = -(y - y_pred).dot(X) + self.regularizacion.gradiente(self.w)
            self.w -= self.tasa_aprendizaje * gradiente

    def predecir(self, X):
        X = np.insert(X, 0, 1, axis=1)
        return X.dot(self.w)

class RegresionLineal(RegresionBase):
    """Modelo de regresión lineal con opción de descenso de gradiente o mínimos cuadrados"""
    def __init__(self, n_iteraciones=100, tasa_aprendizaje=0.001, descenso_gradiente=True):
        self.descenso_gradiente = descenso_gradiente
        self.regularizacion = lambda x: 0
        self.regularizacion.gradiente = lambda x: 0
        super().__init__(n_iteraciones, tasa_aprendizaje)

    def ajustar(self, X, y):
        if not self.descenso_gradiente:
            X = np.insert(X, 0, 1, axis=1)
            U, S, V = np.linalg.svd(X.T.dot(X))
            self.w = V.dot(np.linalg.pinv(np.diag(S))).dot(U.T).dot(X.T).dot(y)
        else:
            super().ajustar(X, y)

class RegresionLasso(RegresionBase):
    """Regresión Lasso con regularización L1 para selección de características"""
    def __init__(self, grado, factor_reg, n_iteraciones=3000, tasa_aprendizaje=0.01):
        self.grado = grado
        self.regularizacion = RegularizacionL1(alpha=factor_reg)
        super().__init__(n_iteraciones, tasa_aprendizaje)

    def ajustar(self, X, y):
        X = normalize(polynomial_features(X, degree=self.grado))
        super().ajustar(X, y)

    def predecir(self, X):
        X = normalize(polynomial_features(X, degree=self.grado))
        return super().predecir(X)

class RegresionPolinomial(RegresionBase):
    """Regresión polinomial para modelar relaciones no lineales"""
    def __init__(self, grado, n_iteraciones=3000, tasa_aprendizaje=0.001):
        self.grado = grado
        self.regularizacion = lambda x: 0
        self.regularizacion.gradiente = lambda x: 0
        super().__init__(n_iteraciones, tasa_aprendizaje)

    def ajustar(self, X, y):
        X = polynomial_features(X, degree=self.grado)
        super().ajustar(X, y)

    def predecir(self, X):
        X = polynomial_features(X, degree=self.grado)
        return super().predecir(X)

class RegresionRidge(RegresionBase):
    """Regresión Ridge con regularización L2 para reducir sobreajuste"""
    def __init__(self, factor_reg, n_iteraciones=1000, tasa_aprendizaje=0.001):
        self.regularizacion = RegularizacionL2(alpha=factor_reg)
        super().__init__(n_iteraciones, tasa_aprendizaje)

class RegresionPolinomialRidge(RegresionBase):
    """Regresión polinomial con regularización Ridge"""
    def __init__(self, grado, factor_reg, n_iteraciones=3000, tasa_aprendizaje=0.01):
        self.grado = grado
        self.regularizacion = RegularizacionL2(alpha=factor_reg)
        super().__init__(n_iteraciones, tasa_aprendizaje)

    def ajustar(self, X, y):
        X = normalize(polynomial_features(X, degree=self.grado))
        super().ajustar(X, y)

    def predecir(self, X):
        X = normalize(polynomial_features(X, degree=self.grado))
        return super().predecir(X)

class ElasticNet(RegresionBase):
    """Modelo ElasticNet que combina regularizaciones L1 y L2"""
    def __init__(self, grado=1, factor_reg=0.05, ratio_l1=0.5, n_iteraciones=3000, tasa_aprendizaje=0.01):
        self.grado = grado
        self.regularizacion = RegularizacionElasticNet(alpha=factor_reg, ratio_l1=ratio_l1)
        super().__init__(n_iteraciones, tasa_aprendizaje)

    def ajustar(self, X, y):
        X = normalize(polynomial_features(X, degree=self.grado))
        super().ajustar(X, y)

    def predecir(self, X):
        X = normalize(polynomial_features(X, degree=self.grado))
        return super().predecir(X)