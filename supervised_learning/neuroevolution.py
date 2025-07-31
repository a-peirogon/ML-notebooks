from __future__ import print_function, division
import numpy as np
import copy

class Neuroevolucion():
    """Optimización evolutiva de Redes Neuronales.

    Parámetros:
    -----------
    tamano_poblacion: int
        Número de redes neuronales en la población.
    tasa_mutacion: float
        Probabilidad de mutación de los pesos.
    constructor_modelo: método
        Método que devuelve una instancia de red neuronal personalizada.
    """
    
    def __init__(self, tamano_poblacion, tasa_mutacion, constructor_modelo):
        self.tamano_poblacion = tamano_poblacion
        self.tasa_mutacion = tasa_mutacion
        self.constructor_modelo = constructor_modelo

    def _crear_modelo(self, id):
        """Crea un nuevo individuo de la población."""
        modelo = self.constructor_modelo(
            n_entradas=self.X.shape[1], 
            n_salidas=self.y.shape[1]
        )
        modelo.id = id
        modelo.aptitud = 0
        modelo.precisión = 0
        return modelo

    def _inicializar_poblacion(self):
        """Inicializa la población de redes neuronales."""
        self.poblacion = [self._crear_modelo(id=np.random.randint(1000)) 
                         for _ in range(self.tamano_poblacion)]

    def _mutar(self, individuo, varianza=1):
        """Aplica mutación gaussiana a los pesos con probabilidad tasa_mutacion."""
        for capa in individuo.capas:
            if hasattr(capa, 'W'):
                # Máscara de mutación para pesos
                mascara = np.random.binomial(1, p=self.tasa_mutacion, size=capa.W.shape)
                capa.W += np.random.normal(loc=0, scale=varianza, size=capa.W.shape) * mascara
                
                # Mutación para pesos del sesgo
                mascara = np.random.binomial(1, p=self.tasa_mutacion, size=capa.w0.shape)
                capa.w0 += np.random.normal(loc=0, scale=varianza, size=capa.w0.shape) * mascara
        return individuo

    def _heredar_pesos(self, hijo, padre):
        """Copia los pesos del padre al hijo."""
        for i in range(len(hijo.capas)):
            if hasattr(hijo.capas[i], 'W'):
                hijo.capas[i].W = padre.capas[i].W.copy()
                hijo.capas[i].w0 = padre.capas[i].w0.copy()

    def _recombinar(self, padre1, padre2):
        """Realiza recombinación entre padres para producir descendencia."""
        hijo1 = self._crear_modelo(id=padre1.id+1)
        self._heredar_pesos(hijo1, padre1)
        
        hijo2 = self._crear_modelo(id=padre2.id+1)
        self._heredar_pesos(hijo2, padre2)

        # Punto de corte aleatorio para la recombinación
        for i in range(len(hijo1.capas)):
            if hasattr(hijo1.capas[i], 'W'):
                n_neuronas = hijo1.capas[i].W.shape[1]
                punto_corte = np.random.randint(0, n_neuronas)
                
                # Intercambio de pesos después del punto de corte
                hijo1.capas[i].W[:, punto_corte:] = padre2.capas[i].W[:, punto_corte:].copy()
                hijo1.capas[i].w0[:, punto_corte:] = padre2.capas[i].w0[:, punto_corte:].copy()
                
                hijo2.capas[i].W[:, punto_corte:] = padre1.capas[i].W[:, punto_corte:].copy()
                hijo2.capas[i].w0[:, punto_corte:] = padre1.capas[i].w0[:, punto_corte:].copy()
        
        return hijo1, hijo2

    def _calcular_aptitud(self):
        """Evalúa las redes neuronales para obtener puntuaciones de aptitud."""
        for individuo in self.poblacion:
            perdida, precision = individuo.test_on_batch(self.X, self.y)
            individuo.aptitud = 1 / (perdida + 1e-8)  # Evita división por cero
            individuo.precisión = precision

    def evolucionar(self, X, y, n_generaciones):
        """Ejecuta el proceso evolutivo durante n_generaciones.
        
        Args:
            X: Datos de entrenamiento
            y: Etiquetas
            n_generaciones: Número de generaciones a evolucionar
            
        Returns:
            El mejor individuo encontrado
        """
        self.X, self.y = X, y
        self._inicializar_poblacion()

        # Configuración de la selección
        n_ganadores = int(self.tamano_poblacion * 0.4)  # 40% mejores
        n_padres = self.tamano_poblacion - n_ganadores  # 60% restantes como padres

        for generacion in range(n_generaciones):
            self._calcular_aptitud()
            
            # Ordenar por aptitud (descendente)
            self.poblacion.sort(key=lambda x: x.aptitud, reverse=True)
            
            mejor_individuo = self.poblacion[0]
            print(f"[Generación {generacion} - Mejor Aptitud: {mejor_individuo.aptitud:.5f}, Precisión: {100*mejor_individuo.precisión:.1f}%]")

            # Nueva población: ganadores + descendencia
            nueva_poblacion = self.poblacion[:n_ganadores]
            
            # Selección de padres proporcional a la aptitud
            aptitud_total = sum(ind.aptitud for ind in self.poblacion)
            probabilidades = [ind.aptitud/aptitud_total for ind in self.poblacion]
            padres = np.random.choice(
                self.poblacion, 
                size=n_padres, 
                p=probabilidades, 
                replace=False
            )
            
            # Recombinación y mutación
            for i in range(0, len(padres), 2):
                hijo1, hijo2 = self._recombinar(padres[i], padres[i+1])
                nueva_poblacion.extend([self._mutar(hijo1), self._mutar(hijo2)])
            
            self.poblacion = nueva_poblacion

        return self.poblacion[0]  # Devuelve el mejor individuo