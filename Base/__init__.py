"""
Paquete Base de Modelo_Ajustes

Contiene módulos y clases reutilizables para manejo de datos, gráficos, ajustes
y funciones auxiliares.

## Clases principales expuestas
    - DF: DataFrame con manejo extendido de datos.
    - G: Graficos para creación y renderizado de figuras.
    - FR: FitResult para manejo de resultados de ajuste.
    - F: Funciones auxiliares y lógica de los ajustes.
"""
from . import Data, Funciones, FitResult, Graficos
from .Data import DataFrame as DF
from .Funciones import Funciones as F
from .FitResult import FitResult as FR
from .Graficos import Graficos as G

__all__ = ["DF", "F", "FR", "G"]