"""
Módulo de resultados de ajuste por regresión ortogonal (ODR).

Este módulo define la clase `FitResult`, que encapsula los resultados de un ajuste
usando Orthogonal Distance Regression (ODR) mediante scipy.odr. Proporciona almacenamiento 
de parámetros ajustados con incertidumbre, coeficientes de determinación, residuos, 
y utilidades para análisis estadístico como el método jackknife para estimar incertidumbres.

## Dependencias
    - scipy.odr.Output : estructura de salida de ajustes ODR.
    - uncertainties.ufloat : para manejar valores con incertidumbre.
    - numpy : para manipulación numérica de arreglos.
    - Decoradores personalizados para gestión de excepciones.

## Uso típico
    Se utiliza como resultado retornado por funciones o clases que implementan el ajuste ODR, 
    permitiendo análisis posterior de la calidad del ajuste y evaluación robusta de incertidumbres.

## Ejemplo simple
    >>> resultado = modelo.fit_odr(x_data, y_data, p0)
    >>> print(resultado)
    >>> parametros_jk, fits_jk = resultado.jackknife(modelo, x_data, y_data)
"""
from ._decoradores import excepciones
from dataclasses import dataclass
from typing import Callable, List, Optional
from uncertainties import ufloat
from scipy.odr import Output
import numpy as np

@dataclass
class FitResult:
    """
    Contenedor de resultados para un ajuste usando Orthogonal Distance Regression (ODR).

    ## Atributos
        - ODR_output scipy.odr.Output:Objeto que contiene toda la información de salida del ajuste ODR.
        - parametros List[ufloat]:Lista de parámetros ajustados con sus valores nominales y desviaciones estándar (incertidumbre).
        - R2 Optional[float]: por defecto None Coeficiente de determinación (R²) para evaluar la calidad del ajuste.
        - R2_aj Optional[float]: por defecto None Coeficiente de determinación ajustado para el número de parámetros y muestras.
        - residuos Optional[np.ndarray]: por defecto None Residuos del ajuste calculados como (y_observado - y_ajustado).
    """
    ODR_output: Output                           # Objeto de salida del ajuste ODR
    parametros: List[ufloat]                        # Parámetros ajustados con incertidumbre
    R2: Optional[float] = None                      # Coeficiente de determinación
    R2_aj: Optional[float] = None                   # R2 ajustado
    residuos: Optional[np.ndarray] = None           # Residuos del ajuste (y_exp - f(*p_opt, x_exp))

    def __str__(self) -> str:
        """
        Representación legible en formato texto del resultado del ajuste.

        Muestra los parámetros ajustados con incertidumbres, valores de R², 
        R² ajustado y motivo(s) de finalización del proceso de ajuste.

        ##  Retorna
            str: Resumen formateado del resultado del ajuste.
        """
        params_lines = "\n".join(
            f"      - p{i+1} = {p.nominal_value:.4g} ± {p.std_dev:.2g}"
            for i, p in enumerate(self.parametros)
        )
        r2_str = f"R² = {self.R2:.4f}" if self.R2 is not None else "R² = N/A"
        r2aj_str = f"R² ajustado = {self.R2_aj:.4f}" if self.R2_aj is not None else "R² ajustado = N/A"
        stopreason_raw = getattr(self.ODR_output, "stopreason", None) if self.ODR_output else None
        if isinstance(stopreason_raw, (list, tuple)):
            stop_lines = "\n".join(f"      - {r}" for r in stopreason_raw)
        elif isinstance(stopreason_raw, str):
            stop_lines = f"      - {stopreason_raw}"
        else:
            stop_lines = "      - N/A"
        content = (
            f"* Parámetros:\n{params_lines}\n"
            f"* {r2_str}\n"
            f"* {r2aj_str}\n"
            f"* Motivo(s) de finalización:\n{stop_lines}"
        )
        border = "#" * 32
        title = "####  Resultado del ajuste  ####"
        return f"{border}\n{title}\n{border}\n{content}\n{border}"
    
    def __iter__(self):
        """
        Iterador que permite desempaquetar los atributos principales del resultado.

        ## Yields
            - List[ufloat]: Parámetros ajustados con incertidumbre.
            - Optional[float]: Coeficiente de determinación R².
            - Optional[float]: Coeficiente de determinación ajustado R²_aj.
            - Optional[np.ndarray]: Residuos del ajuste.
            - scipy.odr.Output: Objeto de salida completo del ajuste ODR.
        """
        yield self.parametros
        yield self.R2
        yield self.R2_aj
        yield self.residuos
        yield self.ODR_output

    @excepciones(critico=True, imprimir=True)
    def jackknife(self, f, data_x: np.ndarray, data_y: np.ndarray, p0: list[float]=None, err_x: Optional[np.ndarray]=None, err_y: Optional[np.ndarray]=None, estimadores: Optional[bool]=True, sstol: Optional[float]=None, partol: Optional[float]=None, maxit: Optional[int]=None, iprint: Optional[int]=None, excluir: Optional[List[int]] = None, incluir: Optional[List[int]] = None) -> List[ufloat]:
        """
        Estima las incertidumbres en los parámetros del ajuste mediante el método estadístico Jackknife.

        Args:
            - f objeto con método callable `fit_odr`: Modelo o función que implementa el método fit_odr para realizar ajustes ODR.
            - data_x np.ndarray: Datos independientes del experimento.
            - data_y np.ndarray: Datos dependientes observados.
            -p0 Optional[List[float]], opcional: Estimación inicial de parámetros para el ajuste (default es None).
            -err_x Optional[np.ndarray], opcional: Incertidumbres en los datos independientes (default es None).
            -err_y Optional[np.ndarray], opcional: Incertidumbres en los datos dependientes (default es None).
            -estimadores Optional[bool], opcional: Flag para controlar cálculo de estimadores en fit_odr (default True).
            -sstol Optional[float], opcional: Tolerancia para convergencia del ajuste (default None).
            -partol Optional[float], opcional: Tolerancia para parámetros en ajuste (default None).
            -maxit Optional[int], opcional: Número máximo de iteraciones del algoritmo ODR (default None).
            -iprint Optional[int], opcional: Nivel de verbosidad del ajuste (default None).
            -excluir Optional[List[int]], opcional: Índices de subconjuntos a excluir del jackknife (default None).
            -incluir Optional[List[int]], opcional: Índices de subconjuntos a incluir exclusivamente (default None).

        Returns:
            - List[ufloat]: Lista de parámetros ajustados con sus incertidumbres estimadas por Jackknife.

        Raises:
        - ValueError: Si se pasan simultáneamente argumentos exclur e incluir.
        - TypeError: Si el objeto `f` no tiene un método callable `fit_odr`.
        - RuntimeError: Si no se logra realizar ningún ajuste durante el proceso Jackknife.
        """
        if excluir and incluir:
            raise ValueError(f"No se pueden pasar ambos argumentos 'incluir' y 'excluir' simultáneamente.")
   
        if not hasattr(f, "fit_odr") or not callable(getattr(f, "fit_odr")):
            raise TypeError("El objeto f debe tener un método callable 'fit_odr'.")

        est_originales = [p.nominal_value for p in self.parametros]
        if not p0 :
            if not self.parametros:
                raise ValueError("No hay parámetros originales para inferir p0.")
            p0 = est_originales

        n = len(data_x)
        fits = []
        mask = np.ones(n, dtype=bool)               # Mascara booleana para excluir los datos i-ésimos
        excluir_set = set(excluir) if excluir is not None else set()
        incluir_set = set(incluir) if incluir is not None else set()

        for i in range(n):
            if incluir_set and i not in incluir_set:
                print(f"Se omite el subconjunto {i+1}.")
                continue
            if i in excluir_set:
                print(f"Se excluye el subconjunto {i+1}.")
                continue

            mask[i] = False                         # Excluye el i-ésimo dato de la mascara          
            x_i, y_i = data_x[mask], data_y[mask]
            ex_i = err_x[mask] if err_x is not None else None
            ey_i = err_y[mask] if err_y is not None else None
            mask[i] = True                          # Retornar el i-ésimo dato de la mascara para la siguiente iteración
            
            try:
                fit_res = f.fit_odr(x_i, y_i, p0, err_x=ex_i, err_y=ey_i, estimadores=estimadores, sstol = sstol, partol = partol, maxit = maxit, iprint = iprint)
            except Exception as e:
                print(f"Se omitió subconjunto {i} por: {e}")
                continue
            fits.append(fit_res)

        if not fits:
            raise RuntimeError("No se pudo completar ningún ajuste en jackknife.")

        estimadores_jk = np.array([[p.nominal_value for p in fr.parametros] for fr in fits])

        medias = np.mean(estimadores_jk, axis=0)
        jack = n * np.array(est_originales) - (n - 1) * medias
        stds = np.sqrt((n - 1) / n * np.sum((estimadores_jk - medias)** 2, axis=0))

        return [ufloat(jack, stds) for jack, stds in zip(jack, stds)], fits