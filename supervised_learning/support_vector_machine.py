from __future__ import division, print_function
import numpy as np
import cvxopt
from MLearning.utils import train_test_split, normalize, accuracy_score
from MLearning.utils.kernels import *
from MLearning.utils import Plot

# Ocultar salida de cvxopt
cvxopt.solvers.options['show_progress'] = False

class MaquinaVectoresSoporte():
    """Clasificador Máquina de Vectores de Soporte (SVM).
    Utiliza cvxopt para resolver el problema de optimización cuadrática.

    Parámetros:
    -----------
    C: float
        Término de penalización para márgenes blandos.
    kernel: function
        Función kernel. Puede ser polinomial, rbf (gaussiano) o lineal.
    grado: int
        Grado del kernel polinomial. Ignorado por otros kernels.
    gamma: float
        Parámetro del kernel rbf.
    coef: float
        Término independiente del kernel polinomial.
    """
    
    def __init__(self, C=1, kernel=kernel_rbf, grado=4, gamma=None, coef=4):
        self.C = C
        self.kernel = kernel
        self.grado = grado
        self.gamma = gamma
        self.coef = coef
        self.multiplicadores_lagrange = None
        self.vectores_soporte = None
        self.etiquetas_vectores_soporte = None
        self.intercepto = None

    def ajustar(self, X, y):
        """Entrena el modelo SVM con los datos proporcionados.
        
        Args:
            X: Matriz de características de entrenamiento
            y: Vector de etiquetas (-1 o 1)
        """
        n_muestras, n_caracteristicas = np.shape(X)

        # Valor por defecto para gamma
        if not self.gamma:
            self.gamma = 1 / n_caracteristicas

        # Configurar función kernel con parámetros
        self.kernel = self.kernel(
            grado=self.grado,
            gamma=self.gamma,
            coef=self.coef)

        # Calcular matriz kernel
        matriz_kernel = np.zeros((n_muestras, n_muestras))
        for i in range(n_muestras):
            for j in range(n_muestras):
                matriz_kernel[i, j] = self.kernel(X[i], X[j])

        # Definir problema de optimización cuadrática
        P = cvxopt.matrix(np.outer(y, y) * matriz_kernel, tc='d')
        q = cvxopt.matrix(np.ones(n_muestras) * -1)
        A = cvxopt.matrix(y, (1, n_muestras), tc='d')
        b = cvxopt.matrix(0, tc='d')

        if not self.C:
            G = cvxopt.matrix(np.identity(n_muestras) * -1)
            h = cvxopt.matrix(np.zeros(n_muestras))
        else:
            G_max = np.identity(n_muestras) * -1
            G_min = np.identity(n_muestras)
            G = cvxopt.matrix(np.vstack((G_max, G_min)))
            h_max = cvxopt.matrix(np.zeros(n_muestras))
            h_min = cvxopt.matrix(np.ones(n_muestras) * self.C)
            h = cvxopt.matrix(np.vstack((h_max, h_min)))

        # Resolver el problema de optimización
        solucion = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Multiplicadores de Lagrange
        mult_lagrange = np.ravel(solucion['x'])

        # Extraer vectores de soporte
        # Índices de multiplicadores no cero
        idx = mult_lagrange > 1e-7
        self.multiplicadores_lagrange = mult_lagrange[idx]
        self.vectores_soporte = X[idx]
        self.etiquetas_vectores_soporte = y[idx]

        # Calcular intercepto con el primer vector de soporte
        self.intercepto = self.etiquetas_vectores_soporte[0]
        for i in range(len(self.multiplicadores_lagrange)):
            self.intercepto -= self.multiplicadores_lagrange[i] * \
                             self.etiquetas_vectores_soporte[i] * \
                             self.kernel(self.vectores_soporte[i], self.vectores_soporte[0])

    def predecir(self, X):
        """Realiza predicciones sobre nuevos datos.
        
        Args:
            X: Matriz de características a predecir
            
        Returns:
            array: Vector de predicciones (-1 o 1)
        """
        y_pred = []
        for muestra in X:
            prediccion = 0
            # Calcular predicción basada en vectores de soporte
            for i in range(len(self.multiplicadores_lagrange)):
                prediccion += self.multiplicadores_lagrange[i] * \
                            self.etiquetas_vectores_soporte[i] * \
                            self.kernel(self.vectores_soporte[i], muestra)
            prediccion += self.intercepto
            y_pred.append(np.sign(prediccion))
        return np.array(y_pred)