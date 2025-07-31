from __future__ import print_function, division
import numpy as np
from scipy.stats import chi2, multivariate_normal
from MLearning.utils import mean_squared_error, train_test_split, polynomial_features

class RegresionBayesiana(object):
    """Modelo de regresión bayesiana. Si se especifica poly_degree, las características
    se transformarán con una función base polinomial, permitiendo regresión polinomial.
    Asume distribución Normal a priori y verosimilitud para los pesos, y distribución
    chi-cuadrado inversa escalada a priori y verosimilitud para la varianza de los pesos.

    Parámetros:
    -----------
    n_draws: float
        Número de muestras simuladas de la posterior de los parámetros.
    mu0: array
        Media de la distribución Normal a priori de los parámetros.
    omega0: array
        Matriz de precisión de la distribución Normal a priori.
    nu0: float
        Grados de libertad de la distribución chi-cuadrado inversa escalada a priori.
    sigma_sq0: float
        Parámetro de escala de la distribución chi-cuadrado inversa escalada a priori.
    poly_degree: int
        Grado polinomial para transformar las características. Permite regresión polinomial.
    cred_int: float
        Intervalo creíble (ETI en esta impl.). 95 => intervalo creíble del 95% para la posterior
        de los parámetros.
    """
    def __init__(self, n_draws, mu0, omega0, nu0, sigma_sq0, poly_degree=0, cred_int=95):
        self.w = None
        self.n_draws = n_draws
        self.poly_degree = poly_degree
        self.cred_int = cred_int

        # Parámetros a priori
        self.mu0 = mu0
        self.omega0 = omega0
        self.nu0 = nu0
        self.sigma_sq0 = sigma_sq0

    # Genera muestras de la distribución chi-cuadrado inversa escalada
    def _muestra_chi_cuadrado_inv_escalada(self, n, df, scale):
        X = chi2.rvs(size=n, df=df)
        sigma_sq = df * scale / X
        return sigma_sq

    def ajustar(self, X, y):
        # Transformación polinomial si es necesario
        if self.poly_degree:
            X = polynomial_features(X, degree=self.poly_degree)

        n_muestras, n_caracteristicas = np.shape(X)
        X_X = X.T.dot(X)

        # Aproximación MCO de beta
        beta_hat = np.linalg.pinv(X_X).dot(X.T).dot(y)

        # Los parámetros posteriores se pueden determinar analíticamente
        # al asumir priores conjugados.

        # Prior Normal / verosimilitud => Posterior Normal
        mu_n = np.linalg.pinv(X_X + self.omega0).dot(X_X.dot(beta_hat)+self.omega0.dot(self.mu0))
        omega_n = X_X + self.omega0
        
        # Prior chi-cuadrado inversa escalada / verosimilitud => Posterior chi-cuadrado inversa escalada
        nu_n = self.nu0 + n_muestras
        sigma_sq_n = (1.0/nu_n)*(self.nu0*self.sigma_sq0 + \
            (y.T.dot(y) + self.mu0.T.dot(self.omega0).dot(self.mu0) - mu_n.T.dot(omega_n.dot(mu_n))))

        # Simular valores de los parámetros
        beta_draws = np.empty((self.n_draws, n_caracteristicas))
        for i in range(self.n_draws):
            sigma_sq = self._muestra_chi_cuadrado_inv_escalada(n=1, df=nu_n, scale=sigma_sq_n)
            beta = multivariate_normal.rvs(size=1, mean=mu_n[:,0], cov=sigma_sq*np.linalg.pinv(omega_n))
            beta_draws[i, :] = beta

        # Usar la media de las simulaciones para predicciones
        self.w = np.mean(beta_draws, axis=0)

        # Límites del intervalo creíble
        l_eti = 50 - self.cred_int/2
        u_eti = 50 + self.cred_int/2
        self.eti = np.array([[np.percentile(beta_draws[:,i], q=l_eti), np.percentile(beta_draws[:,i], q=u_eti)] \
                                for i in range(n_caracteristicas)])

    def predecir(self, X, eti=False):
        # Transformación polinomial si es necesario
        if self.poly_degree:
            X = polynomial_features(X, degree=self.poly_degree)

        y_pred = X.dot(self.w)
        # Devolver límites del intervalo creíble si se solicita
        if eti:
            lower_w = self.eti[:, 0]
            upper_w = self.eti[:, 1]
            y_lower_pred = X.dot(lower_w)
            y_upper_pred = X.dot(upper_w)
            return y_pred, y_lower_pred, y_upper_pred

        return y_pred