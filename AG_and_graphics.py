#!/usr/bin/env python3
import os
import random
import sys
from datetime import datetime  # Add this import

# Verificar si estamos en un entorno virtual
if not hasattr(sys, 'real_prefix') and not sys.base_prefix != sys.prefix:
    print("ADVERTENCIA: No se detectó un entorno virtual.")
    print("Se recomienda usar un entorno virtual para este programa.")
    print("Cree uno con: python3 -m venv venv")
    print("Y actívelo con: source venv/bin/activate")
    print("Luego instale las dependencias con: pip install numpy matplotlib seaborn")
    print("\nContinuando de todas formas...\n")

print(f"Usando Python desde: {sys.executable}")  # Para verificar qué Python estamos usando

NUMPY_AVAILABLE = False
SEABORN_AVAILABLE = False

# Reemplazar la configuración del backend con algo más simple y robusto
import matplotlib
matplotlib.use('Agg')  # Usar backend no interactivo por defecto
import matplotlib.pyplot as plt

# Configurar backend antes de importar pyplot
interactive_backend = False

try:
    import numpy as np
    from typing import List, Tuple
    
    # Verificar numpy y matplotlib
    test_array = np.array([1, 2, 3])
    NUMPY_AVAILABLE = True
    
    # Verificar e inicializar seaborn de manera más robusta
    try:
        import seaborn as sns
        print("Importando seaborn desde:", sns.__file__)  # Para diagnóstico
        # Configuración básica de seaborn
        sns.set_theme(style="whitegrid", context="notebook")
        print("Seaborn inicializado correctamente")
        SEABORN_AVAILABLE = True
    except ImportError as e:
        print(f"Error importando seaborn: {e}")
        print(f"Python path: {sys.path}")  # Para diagnóstico
        plt.style.use('default')
        
except (ImportError, AttributeError) as e:
    print(f"Error al cargar las dependencias básicas: {str(e)}")
    print("Por favor, instale o actualice las dependencias con:")
    print(f"{sys.executable} -m pip install numpy matplotlib seaborn --upgrade")
    # Usar alternativas básicas de Python
    from array import array
    from typing import List, Tuple

# Modificar la clase Individuo para tener una función de aptitud más interesante
class Individuo:
    def __init__(self, longitud_cromosoma: int = 8):
        if longitud_cromosoma <= 0:
            raise ValueError("La longitud del cromosoma debe ser positiva")
        if not NUMPY_AVAILABLE:
            self.genes = [random.randint(0, 1) for _ in range(longitud_cromosoma)]
        else:
            self.genes = np.random.randint(2, size=longitud_cromosoma)
        self.aptitud = 0

    def calcular_aptitud(self) -> float:
        """Calcula y retorna la aptitud del individuo usando una función más compleja"""
        if NUMPY_AVAILABLE:
            # Función objetivo: maximizar 1s consecutivos
            grupos_unos = np.split(self.genes, np.where(np.diff(self.genes) != 0)[0] + 1)
            valores_grupos = [sum(grupo) ** 2 if grupo[0] == 1 else 0 for grupo in grupos_unos]
            self.aptitud = sum(valores_grupos)
        else:
            # Versión simple para cuando no está numpy
            self.aptitud = sum(self.genes)
        return self.aptitud

class AlgoritmoGenetico:
    def __init__(self, tamanio_poblacion: int = 50, longitud_cromosoma: int = 8, 
                 tasa_mutacion: float = 0.01, tasa_mutacion_max: float = 0.1):
        if tamanio_poblacion <= 0 or longitud_cromosoma <= 0:
            raise ValueError("Tamaño de población y longitud de cromosoma deben ser positivos")
        if not 0 <= tasa_mutacion <= 1:
            raise ValueError("La tasa de mutación debe estar entre 0 y 1")
        
        self.tamanio_poblacion = tamanio_poblacion
        self.longitud_cromosoma = longitud_cromosoma
        self.tasa_mutacion = tasa_mutacion
        self.tasa_mutacion_max = tasa_mutacion_max
        self.generaciones_sin_mejora = 0
        self.mejor_aptitud_historica = float('-inf')
        self.poblacion = [Individuo(longitud_cromosoma) for _ in range(tamanio_poblacion)]
        self.mejor_historico = None
        self.historial_convergencia = []
        self.diversidad_minima = 0.2
        self.tasa_inmigracion = 0.1
        self.ultimo_mejor = float('-inf')
        self.poblacion_referencia = None
        self.contador_estancamiento = 0
        self.memoria_elite = []
        self.max_memoria_elite = 10
        self.radio_nicho = 0.2
        self.tasa_cruce = 0.7
        self.especies = []
        self.num_especies = 5
    
    def calcular_diversidad(self) -> float:
        """Calcula la diversidad de la población usando distancia de Hamming promedio"""
        if not NUMPY_AVAILABLE:
            total_diff = 0
            for i in range(len(self.poblacion)):
                for j in range(i + 1, len(self.poblacion)):
                    total_diff += sum(g1 != g2 for g1, g2 in zip(
                        self.poblacion[i].genes, self.poblacion[j].genes))
            return total_diff / (len(self.poblacion) * (len(self.poblacion) - 1) / 2)
        else:
            genes_array = np.array([ind.genes for ind in self.poblacion])
            return np.mean([np.mean(np.abs(genes_array - genes_array[i]))
                          for i in range(len(genes_array))])

    def busqueda_local(self, individuo: Individuo) -> None:
        """Aplica búsqueda local adaptativa al individuo"""
        aptitud_original = individuo.calcular_aptitud()
        mejora = False
        
        for i in range(len(individuo.genes)):
            individuo.genes[i] = 1 - individuo.genes[i]  # Flip bit
            nueva_aptitud = individuo.calcular_aptitud()
            
            if nueva_aptitud <= aptitud_original:
                individuo.genes[i] = 1 - individuo.genes[i]  # Revertir cambio
            else:
                mejora = True
                aptitud_original = nueva_aptitud
        
        return mejora

    def aplicar_inmigracion(self) -> None:
        """Introduce nuevos individuos aleatorios en la población"""
        num_inmigrantes = int(self.tamanio_poblacion * self.tasa_inmigracion)
        indices = random.sample(range(self.tamanio_poblacion), num_inmigrantes)
        
        for idx in indices:
            self.poblacion[idx] = Individuo(self.longitud_cromosoma)

    def reinicio_parcial(self, porcentaje: float = 0.2) -> None:
        """Reinicio parcial mejorado con preservación de diversidad"""
        aptitudes = [ind.calcular_aptitud() for ind in self.poblacion]
        indices_ordenados = sorted(range(len(aptitudes)), key=lambda k: aptitudes[k], reverse=True)
        
        num_mantener = int(self.tamanio_poblacion * (1 - porcentaje))
        poblacion_elite = [self.poblacion[i] for i in indices_ordenados[:num_mantener]]
        
        nuevos_individuos = [Individuo(self.longitud_cromosoma) 
                           for _ in range(self.tamanio_poblacion - num_mantener)]
        self.poblacion = poblacion_elite + nuevos_individuos

    def ajustar_tasa_mutacion(self) -> None:
        """Ajusta la tasa de mutación basada en la diversidad de la población"""
        if self.generaciones_sin_mejora > 10:
            self.tasa_mutacion = min(self.tasa_mutacion * 1.5, self.tasa_mutacion_max)
        else:
            self.tasa_mutacion = max(self.tasa_mutacion * 0.9, 0.01)

    def cruzar(self, padre1: Individuo, padre2: Individuo) -> Individuo:
        """Realiza el cruce de dos padres y retorna un hijo"""
        hijo = Individuo(self.longitud_cromosoma)
        punto = random.randint(0, self.longitud_cromosoma-1)
        if NUMPY_AVAILABLE:
            hijo.genes = np.concatenate((padre1.genes[:punto], padre2.genes[punto:]))
        else:
            hijo.genes = padre1.genes[:punto] + padre2.genes[punto:]
        return hijo
    
    def cruce_uniforme_adaptativo(self, padre1: Individuo, padre2: Individuo) -> Individuo:
        """Implementa cruce uniforme con probabilidad adaptativa"""
        hijo = Individuo(self.longitud_cromosoma)
        similitud = sum(g1 == g2 for g1, g2 in zip(padre1.genes, padre2.genes)) / self.longitud_cromosoma
        
        # Ajustar probabilidad de cruce basada en similitud
        prob_cruce = self.tasa_cruce * (1.0 - similitud)
        
        if NUMPY_AVAILABLE:
            mascara = np.random.random(self.longitud_cromosoma) < prob_cruce
            hijo.genes = np.where(mascara, padre1.genes, padre2.genes)
        else:
            hijo.genes = [g1 if random.random() < prob_cruce else g2 
                         for g1, g2 in zip(padre1.genes, padre2.genes)]
        return hijo
    
    def actualizar_memoria_elite(self, individuo: Individuo) -> None:
        """Mantiene una memoria de las mejores soluciones encontradas"""
        if not self.memoria_elite or individuo.aptitud > self.memoria_elite[0].aptitud:
            nuevo_ind = Individuo(self.longitud_cromosoma)
            nuevo_ind.genes = individuo.genes.copy()
            nuevo_ind.aptitud = individuo.aptitud
            
            self.memoria_elite.append(nuevo_ind)
            self.memoria_elite.sort(key=lambda x: x.aptitud, reverse=True)
            
            if len(self.memoria_elite) > self.max_memoria_elite:
                self.memoria_elite.pop()

    def calcular_distancia(self, ind1: Individuo, ind2: Individuo) -> float:
        """Calcula la distancia genética entre dos individuos"""
        if NUMPY_AVAILABLE:
            return np.mean(np.abs(np.array(ind1.genes) - np.array(ind2.genes)))
        return sum(g1 != g2 for g1, g2 in zip(ind1.genes, ind2.genes)) / len(ind1.genes)

    def identificar_especies(self) -> None:
        """Agrupa la población en especies usando clustering"""
        self.especies = []
        no_asignados = self.poblacion.copy()
        
        while len(no_asignados) > 0 and len(self.especies) < self.num_especies:
            # Seleccionar representante de nueva especie
            representante = max(no_asignados, key=lambda x: x.aptitud)
            especie = []
            
            # Asignar individuos a la especie
            i = 0
            while i < len(no_asignados):
                if self.calcular_distancia(representante, no_asignados[i]) < self.radio_nicho:
                    especie.append(no_asignados.pop(i))
                else:
                    i += 1
            
            if especie:
                self.especies.append(especie)

    def aplicar_competencia_limitada(self) -> None:
        """Aplica competencia limitada entre especies"""
        for especie in self.especies:
            # Normalizar aptitudes dentro de la especie
            total_aptitud = sum(ind.aptitud for ind in especie)
            if total_aptitud > 0:
                for ind in especie:
                    ind.aptitud = ind.aptitud / len(especie)

    def mutar(self, individuo: Individuo) -> None:
        """Aplica mutación a un individuo"""
        for i in range(len(individuo.genes)):
            if np.random.random() < self.tasa_mutacion:
                individuo.genes[i] = 1 - individuo.genes[i]

    def seleccion_ruleta(self) -> Individuo:
        """Selecciona un individuo usando el método de la ruleta"""
        aptitudes = [ind.calcular_aptitud() for ind in self.poblacion]
        suma_aptitudes = sum(aptitudes)
        
        if suma_aptitudes == 0:
            return random.choice(self.poblacion)
        
        if NUMPY_AVAILABLE:
            probabilidades = np.array(aptitudes) / suma_aptitudes
            indice = np.random.choice(len(self.poblacion), p=probabilidades)
            return self.poblacion[indice]
        else:
            punto = random.uniform(0, suma_aptitudes)
            acumulado = 0
            for i, individuo in enumerate(self.poblacion):
                acumulado += aptitudes[i]
                if acumulado >= punto:
                    return individuo
            return self.poblacion[-1]

    def seleccion_torneo(self, tamano_torneo: int = 3) -> Individuo:
        """Selecciona un individuo usando selección por torneo"""
        if tamano_torneo > len(self.poblacion):
            tamano_torneo = len(self.poblacion)
            
        if NUMPY_AVAILABLE:
            indices = np.random.choice(len(self.poblacion), tamano_torneo, replace=False)
            participantes = [self.poblacion[i] for i in indices]
        else:
            participantes = random.sample(self.poblacion, tamano_torneo)
            
        return max(participantes, key=lambda x: x.calcular_aptitud())
    
    def evolucionar(self, generaciones: int = 100) -> Tuple[List[float], List[float], List[float]]:
        """Versión mejorada del método evolucionar con mejor tracking de convergencia y feedback"""
        historial_mejor = []
        historial_promedio = []
        historial_peor = []
        self.historial_convergencia = []
        
        print(f"\nEvolucionando población de {self.tamanio_poblacion} individuos durante {generaciones} generaciones")
        print("Progreso: ", end='', flush=True)
        
        # Calcular puntos de reporte (cada 10%)
        puntos_reporte = {int(generaciones * i/10): i for i in range(1, 11)}
        ultima_mejora = 0
        
        for gen in range(generaciones):
            # Mostrar progreso
            if gen in puntos_reporte:
                print(f"{puntos_reporte[gen]}0%... ", end='', flush=True)
            
            # Identificar especies y aplicar nichos
            self.identificar_especies()
            self.aplicar_competencia_limitada()
            
            nueva_poblacion = []
            aptitudes = [ind.calcular_aptitud() for ind in self.poblacion]
            mejor_actual = max(aptitudes)
            promedio_actual = sum(aptitudes) / len(aptitudes)
            peor_actual = min(aptitudes)
            
            # Reportar mejoras significativas
            if mejor_actual > self.mejor_aptitud_historica:
                if gen - ultima_mejora > 10:  # Evitar demasiados mensajes
                    print(f"\nMejora en gen {gen}: {mejor_actual:.2f}")
                    ultima_mejora = gen
                self.mejor_aptitud_historica = mejor_actual
                self.generaciones_sin_mejora = 0
            else:
                self.generaciones_sin_mejora += 1
            
            # Selección y cruce
            while len(nueva_poblacion) < self.tamanio_poblacion:
                padre1 = self.seleccion_torneo()
                padre2 = self.seleccion_torneo()
                if random.random() < self.tasa_cruce:
                    hijo = self.cruce_uniforme_adaptativo(padre1, padre2)
                else:
                    hijo = self.cruzar(padre1, padre2)
                self.mutar(hijo)
                nueva_poblacion.append(hijo)
            
            self.poblacion = nueva_poblacion
            
            # Registrar estadísticas
            historial_mejor.append(mejor_actual)
            historial_promedio.append(promedio_actual)
            historial_peor.append(peor_actual)
            
            # Control de diversidad y estancamiento
            if self.generaciones_sin_mejora > 20:
                print(f"\nReinicio parcial en generación {gen}")
                self.reinicio_parcial(0.3)
                self.generaciones_sin_mejora = 0
        
        print("\nEvolución completada!")
        print(f"Mejor aptitud alcanzada: {max(historial_mejor):.2f}")
        print(f"Mejora total: {max(historial_mejor) - historial_mejor[0]:.2f}")
        
        return historial_mejor, historial_promedio, historial_peor

def visualizar_resultados(mejor: List[float], promedio: List[float], peor: List[float]) -> None:
    """Visualiza los resultados de la evolución con gráficos detallados"""
    if not NUMPY_AVAILABLE:
        print("Resultados finales:")
        print(f"Mejor aptitud: {max(mejor)}")
        print(f"Aptitud promedio: {sum(promedio)/len(promedio)}")
        print(f"Peor aptitud: {min(peor)}")
        return

    try:
        plt.close('all')
        
        # Configurar el estilo
        if SEABORN_AVAILABLE:
            sns.set_style("whitegrid")
            sns.set_context("talk")
        
        # Crear figura y subplots
        fig = plt.figure(figsize=(15, 12))
        gs = plt.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])

        # Convertir datos a numpy arrays
        generaciones = np.arange(len(mejor))
        mejor_array = np.array(mejor)
        promedio_array = np.array(promedio)
        peor_array = np.array(peor)

        # 1. Evolución de aptitudes con más detalle
        ax1.plot(generaciones, mejor_array, 'g-', label='Mejor', linewidth=2)
        ax1.plot(generaciones, promedio_array, 'b-', label='Promedio', linewidth=1.5)
        ax1.plot(generaciones, peor_array, 'r-', label='Peor', linewidth=1.5, alpha=0.7)
        ax1.set_xlabel('Generación')
        ax1.set_ylabel('Aptitud')
        ax1.set_title('Evolución de la Aptitud')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)

        # 2. Convergencia y Diversidad mejorada
        convergencia = mejor_array - promedio_array
        diversidad = mejor_array - peor_array
        ax2.fill_between(generaciones, convergencia, alpha=0.3, color='blue', label='Área de Convergencia')
        ax2.plot(generaciones, convergencia, 'b-', label='Convergencia', linewidth=2)
        ax2.plot(generaciones, diversidad, 'r--', label='Diversidad', linewidth=2)
        ax2.set_xlabel('Generación')
        ax2.set_ylabel('Diferencia de Aptitud')
        ax2.set_title('Convergencia y Diversidad')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)

        # 3. Distribución de aptitudes mejorada
        if SEABORN_AVAILABLE:
            sns.violinplot(data=[mejor_array, promedio_array, peor_array], ax=ax3)
        else:
            ax3.boxplot([mejor_array, promedio_array, peor_array])
        ax3.set_xticklabels(['Mejor', 'Promedio', 'Peor'])
        ax3.set_title('Distribución de Aptitudes')
        ax3.set_ylabel('Aptitud')
        ax3.grid(True, alpha=0.3)

        # 4. Análisis de mejora con suavizado
        ventana = max(3, len(mejor) // 50)  # Ventana adaptativa para suavizado
        kernel = np.ones(ventana) / ventana
        mejoras_suavizadas = np.convolve(np.diff(mejor_array), kernel, mode='valid')
        x_mejoras = generaciones[ventana:][:len(mejoras_suavizadas)]
        
        ax4.plot(x_mejoras, mejoras_suavizadas, 'g-', label='Tasa de Mejora', linewidth=2)
        ax4.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax4.set_xlabel('Generación')
        ax4.set_ylabel('Mejora')
        ax4.set_title('Análisis de Mejora')
        ax4.legend(loc='best')
        ax4.grid(True, alpha=0.3)

        # Añadir estadísticas generales
        stats_text = (
            f'Estadísticas Finales:\n'
            f'Mejor aptitud: {np.max(mejor_array):.2f}\n'
            f'Aptitud promedio final: {promedio_array[-1]:.2f}\n'
            f'Mejora total: {mejor_array[-1] - mejor_array[0]:.2f}\n'
            f'Generaciones: {len(mejor)}'
        )
        plt.figtext(0.02, 0.02, stats_text, fontsize=10, 
                   bbox=dict(facecolor='white', alpha=0.8))

        # Ajustar layout y mostrar
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Guardar o mostrar según el backend
        if not interactive_backend:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'resultados_ag_{timestamp}.png'
            print(f"Guardando resultados como '{filename}'...")
            plt.savefig(filename, bbox_inches='tight', dpi=300)
        else:
            try:
                plt.show(block=True)
            except Exception as e:
                print(f"Error mostrando gráfica: {e}")
                print("Guardando como respaldo en 'resultados_ag.png'...")
                plt.savefig('resultados_ag.png', bbox_inches='tight', dpi=300)

    except Exception as e:
        print(f"Error en visualización: {e}")
        import traceback
        traceback.print_exc()

# Modificar la función principal para asegurar la generación y visualización
if __name__ == "__main__":
    if NUMPY_AVAILABLE:
        np.random.seed(42)
    random.seed(42)
    
    try:
        print("Iniciando algoritmo genético...")
        
        # Configuración con parámetros más agresivos
        ag = AlgoritmoGenetico(
            tamanio_poblacion=200,     # Reducido para prueba inicial
            longitud_cromosoma=30,     # Reducido para prueba inicial
            tasa_mutacion=0.05,        # Mayor tasa de mutación inicial
            tasa_mutacion_max=0.3      # Mayor tasa de mutación máxima
        )
        
        print("Ejecutando evolución...")
        mejor, promedio, peor = ag.evolucionar(generaciones=100)  # Reducido para prueba inicial
        
        print(f"\nResultados finales:")
        print(f"Mejor aptitud alcanzada: {max(mejor)}")
        print(f"Aptitud promedio final: {promedio[-1]}")
        print(f"Peor aptitud final: {peor[-1]}")
        
        print("\nGenerando visualización...")
        # Siempre guardar en archivo
        plt.figure(figsize=(15, 12))
        plt.plot(mejor, 'g-', label='Mejor')
        plt.plot(promedio, 'b-', label='Promedio')
        plt.plot(peor, 'r-', label='Peor')
        plt.title('Evolución de la Aptitud')
        plt.xlabel('Generación')
        plt.ylabel('Aptitud')
        plt.legend()
        plt.grid(True)
        plt.savefig('evolucion_ag.png')
        plt.close()
        
        # Intentar mostrar visualización completa
        visualizar_resultados(mejor, promedio, peor)
        print("\nVisualización guardada en 'evolucion_ag.png' y 'resultados_ag.png'")
        
    except Exception as e:
        print(f"Error durante la ejecución: {e}")
        import traceback
        traceback.print_exc()