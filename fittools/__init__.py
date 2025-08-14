"""
Paquete principal fittools de Modelo-Ajustes

Contiene módulos y clases reutilizables para manejo de datos, gráficos, ajustes
y funciones auxiliares.

## Clases principales expuestas
    - dframes: DataFrame con manejo extendido de datos.
    - grafs: Graficos para creación y renderizado de figuras.
    - fres: FitResult para manejo de resultados de ajuste.
    - funcs: Funciones auxiliares y lógica de los ajustes.

Módulos internos como _decoradores no se exponen.
"""
from . import funciones, graficos, data_frames, fit_result
from .data_frames import DataFrame as dframes
from .funciones import Funciones as funcs
from .fit_result import FitResult as fres
from .graficos import Graficos as grafs

__all__ = ["dframes", "funcs", "fres", "grafs"]