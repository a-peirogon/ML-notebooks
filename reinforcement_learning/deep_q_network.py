from __future__ import print_function, division
import random
import numpy as np
import gym
from collections import deque


class DeepQNetwork():
    """Aprendizaje Q con red neuronal profunda para aprender políticas de control.
    Utiliza un modelo de red neuronal para predecir la utilidad esperada (valor Q) de ejecutar una acción en un estado dado.

    Referencia: https://arxiv.org/abs/1312.5602
    
    Parámetros:
    -----------
    nombre_entorno: str
        Nombre del entorno de OpenAI Gym a explorar.
        Ver: https://gym.openai.com/envs
    epsilon: float
        Valor para la política epsilon-greedy. Probabilidad de seleccionar una acción aleatoria.
    gamma: float
        Factor de descuento para recompensas futuras.
    tasa_decaida: float
        Tasa de decaimiento del valor epsilon.
    epsilon_min: float
        Valor mínimo que puede tomar epsilon.
    """
    
    def __init__(self, nombre_entorno='CartPole-v1', epsilon=1, gamma=0.9, tasa_decaida=0.005, epsilon_min=0.1):
        self.epsilon = epsilon
        self.gamma = gamma
        self.tasa_decaida = tasa_decaida
        self.epsilon_min = epsilon_min
        self.tamano_memoria = 300
        self.memoria = []

        # Inicializar entorno
        self.entorno = gym.make(nombre_entorno)
        self.n_estados = self.entorno.observation_space.shape[0]
        self.n_acciones = self.entorno.action_space.n
    
    def establecer_modelo(self, modelo):
        """Configura el modelo de red neuronal a utilizar."""
        self.modelo = modelo(n_entradas=self.n_estados, n_salidas=self.n_acciones)

    def _seleccionar_accion(self, estado):
        """Selecciona una acción usando política epsilon-greedy."""
        if np.random.rand() < self.epsilon:
            # Acción aleatoria
            return np.random.randint(self.n_acciones)
        else:
            # Acción con mayor valor Q predicho
            return np.argmax(self.modelo.predecir(estado), axis=1)[0]

    def _memorizar(self, estado, accion, recompensa, nuevo_estado, terminado):
        """Almacena la experiencia en la memoria de repetición."""
        self.memoria.append((estado, accion, recompensa, nuevo_estado, terminado))
        # Mantener tamaño máximo de memoria
        if len(self.memoria) > self.tamano_memoria:
            self.memoria.pop(0)

    def _construir_conjunto_entrenamiento(self, replay):
        """Construye el conjunto de entrenamiento a partir de experiencias almacenadas."""
        estados = np.array([a[0] for a in replay])
        nuevos_estados = np.array([a[3] for a in replay])

        Q = self.modelo.predecir(estados)
        Q_nuevo = self.modelo.predecir(nuevos_estados)

        tamano_replay = len(replay)
        X = np.empty((tamano_replay, self.n_estados))
        y = np.empty((tamano_replay, self.n_acciones))
        
        for i in range(tamano_replay):
            estado_r, accion_r, recompensa_r, nuevo_estado_r, terminado_r = replay[i]

            objetivo = Q[i]
            objetivo[accion_r] = recompensa_r
            if not terminado_r:
                objetivo[accion_r] += self.gamma * np.amax(Q_nuevo[i])

            X[i] = estado_r
            y[i] = objetivo

        return X, y

    def entrenar(self, n_epocas=500, tamano_lote=32):
        """Entrena el agente en el entorno especificado."""
        recompensa_maxima = 0

        for epoca in range(n_epocas):
            estado = self.entorno.reset()
            recompensa_total = 0
            perdidas_epoca = []

            while True:
                accion = self._seleccionar_accion(estado)
                nuevo_estado, recompensa, terminado, _ = self.entorno.step(accion)

                self._memorizar(estado, accion, recompensa, nuevo_estado, terminado)

                # Muestrear lote de memoria
                _tamano_lote = min(len(self.memoria), tamano_lote)
                replay = random.sample(self.memoria, _tamano_lote)

                X, y = self._construir_conjunto_entrenamiento(replay)
                perdida = self.modelo.entrenar_lote(X, y)
                perdidas_epoca.append(perdida)

                recompensa_total += recompensa
                estado = nuevo_estado

                if terminado: break
            
            perdida_promedio = np.mean(perdidas_epoca)
            self.epsilon = self.epsilon_min + (1.0 - self.epsilon_min) * np.exp(-self.tasa_decaida * epoca)
            recompensa_maxima = max(recompensa_maxima, recompensa_total)

            print(f"Época {epoca} [Pérdida: {perdida_promedio:.4f}, Recompensa: {recompensa_total}, Epsilon: {self.epsilon:.4f}, Máx Recompensa: {recompensa_maxima}]")

        print("Entrenamiento completado")

    def jugar(self, n_epocas):
        """Ejecuta el agente entrenado en el entorno."""
        for epoca in range(n_epocas):
            estado = self.entorno.reset()
            recompensa_total = 0
            while True:
                self.entorno.render()
                accion = np.argmax(self.modelo.predecir(estado), axis=1)[0]
                estado, recompensa, terminado, _ = self.entorno.step(accion)
                recompensa_total += recompensa
                if terminado: break
            print(f"Época {epoca} Recompensa: {recompensa_total}")
        self.entorno.close()