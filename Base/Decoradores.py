from typing import Callable
from functools import wraps
import time

def timer(func):
    """
    Decorador para medir y mostrar el tiempo de ejecución de una función o método.

    Este decorador utiliza "time.perf_counter()" para obtener una medición de tiempo,
    adecuada para benchmarking y análisis de rendimiento.

    Args:
        func (callable): La función o método al que se aplica el decorador.

    Returns:
        callable: Una función wrapper que ejecuta la función original, mide su tiempo
                  de ejecución, imprime el tiempo transcurrido en segundos con precisión
                  de microsegundos y devuelve el resultado original.

    ## Uso típico
        ```python
        @medir_tiempo
        def mi_funcion():
            #Código cuya duración queremos medir**
            pass
        ```
    ## Salida
        Tiempo de ejecución de mi_funcion: x segundos
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        inicio = time.perf_counter()
        resultado = func(*args, **kwargs)
        fin = time.perf_counter()
        print(f"Tiempo de ejecución de {func.__name__}: {fin - inicio:.6f} segundos")
        return resultado
    return wrapper

def excepciones(critico: bool = False, imprimir: bool = True):
    """
    Decorador que maneja excepciones en métodos o funciones.

    Args:
    - critico (bool): Si es True, la excepción se propaga y detiene el programa.
                      Si es False, se atrapa, se muestra (si imprimir=True) y se devuelve None.
    - imprimir (bool): Si es True, imprime un mensaje de error. Si es False, es silencioso.

    ## Uso típico
        @excepciones(critico=False)
        def metodo_auxiliar(): ...

        @excepciones(critico=True)
        def metodo_critico(): ...
    """
    def decorador(func: Callable) -> Callable:
        @wraps(func)  # Preserva el nombre y docstring de la función original
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)

            # Captura errores comunes
            except FileNotFoundError as e:
                mensaje = f"[{func.__name__}] Archivo no encontrado: {e}"
                error = e
            except ValueError as e:
                mensaje = f"[{func.__name__}] Valor inválido: {e}"
                error = e
            except Exception as e:
                mensaje = f"[{func.__name__}] Error inesperado: {e}"
                error = e

            # Imprime si está activado
            if imprimir:
                print(mensaje)
            # Si es crítico, volvemos a lanzar la excepción
            if critico:
                raise error
            # Si no es crítico, devolvemos None para que el flujo continúe
            return None
        return wrapper
    return decorador