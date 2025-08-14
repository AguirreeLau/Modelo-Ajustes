"""
Módulo Graficos

Este módulo contiene la clase `Graficos`, que permite crear, configurar y mostrar gráficos
con Matplotlib de manera flexible. Incluye soporte para múltiples columnas, estilos,
colores personalizados, tamaños de fuente y guardado automático de figuras en carpetas.

## Dependencias
    - matplotlib
    - numpy
    - pathlib
    - scienceplots
    - DataFrame (para normalización de strings)
    - decoradores (para manejo de excepciones)
"""
from typing import Optional, Union, Sequence
import matplotlib.pyplot as plt
from matplotlib import style
import scienceplots
from pathlib import Path
import numpy as np
from .data_frames import DataFrame as DF        # para usar _norm_str
from ._decoradores import excepciones
from dataclasses import dataclass, field

@dataclass
class Graficos:
    """
    Clase para crear y mostrar gráficos de manera flexible con Matplotlib.

    ## Atributos
        - columnas (int): Número de columnas de subplots (default: 1)
        - tamaño (tuple[int,int]): Tamaño de la figura en pulgadas (ancho, alto)
        - colores (dict[str,str]): Diccionario de colores para título, ejes, ticks, grilla y fondo
        - fontsizes (dict[str,int]): Diccionario de tamaños de fuente para título, ejes, ticks y leyenda

    ## Uso típico
        ```python
        g = Graficos(columnas=2, colores={"titulo": "#FF0000"})
        g.crear(titulo=["Graf1", "Graf2"])
        g.render(l_cols=[1,2], ruta_guardado="figura.png")
        ```
    """
    columnas: int = 1
    tamaño: tuple[int, int] = (8,5)     # (ancho, alto)
    colores: dict[str,str] = field(default_factory=lambda: {
        "titulo": "#1F2020",
        "eje_x": "#1F2020",
        "eje_y": "#1F2020",
        "ticks": "#1F2020",
        "grilla": "#ACACAC",
        "fondo": "#FFFFFF",
    })
    fontsizes: dict[str,int] = field(default_factory=lambda: {
        "titulo": 14,
        "eje_x": 12,
        "eje_y": 12,
        "ticks": 10,
        "label": 10
    })

    @excepciones(critico=True, imprimir=True)
    def __post_init__(self):    # Combinar fontsizes con defaults
        """
        Inicialización posterior a la creación de la instancia.

        Combina los diccionarios de colores y fontsizes proporcionados con los
        valores predeterminados usando la función `_merge_defaults`.
        """
        self.colores = self._merge_defaults(
            defaults={
                "titulo": "#1F2020",
                "eje_x": "#1F2020",
                "eje_y": "#1F2020",
                "ticks": "#1F2020",
                "grilla": "#ACACAC",
                "fondo": "#FFFFFF"
            },
            custom=self.colores
        )
        self.fontsizes = self._merge_defaults(
            defaults={
                "titulo": 14,
                "eje_x": 12,
                "eje_y": 12,
                "ticks": 10,
                "leyenda": 10
            },
            custom=self.fontsizes
        )

    @staticmethod
    @excepciones(critico=True, imprimir=True)
    def _merge_defaults(defaults: dict, custom: Optional[dict]) -> dict:
        """
        Combina un diccionario de valores por defecto con otro personalizado.

        Args:
            defaults (dict): Diccionario de valores predeterminados.
            custom (dict | None): Diccionario de valores a sobreescribir o None.

        Returns:
            dict: Diccionario combinado donde los valores de `custom` sobreescriben
                  los de `defaults` y las keys son normalizadas usando `_norm_str`.
        """
        if custom is None:
            return defaults
        if not isinstance(custom, dict):
            raise ValueError("El parámetro debe ser un diccionario o None.")

        custom_norm = {DF._norm_str(k): v for k, v in custom.items()}
        # Normalizar keys de defaults
        defaults_norm = {DF._norm_str(k): v for k, v in defaults.items()} 
        return {**defaults_norm, **custom_norm}

    @staticmethod
    @excepciones(critico=True, imprimir=True)
    def _ajustar_lista(val: Optional[Union[int, float, Sequence[Union[int, float]]]], longitud: int) -> list[Optional[Union[int, float]]]:
        """
        Ajusta un valor o secuencia para que tenga la longitud requerida.

        Args:
            val (int | float | Sequence | None): Valor único o secuencia de valores.
            longitud (int): Longitud deseada de la lista de salida.

        Returns:
            list: Lista de longitud `longitud` con los valores ajustados.
        """
        if val is None:
            return [None] * longitud
        if not isinstance(val, (list, tuple, np.ndarray)):
            return [val] * longitud
        if len(val) < longitud:
            return list(val) + [None] * (longitud - len(val))
        elif len(val) > longitud:
            return list(val)[:longitud]
        else:
            return list(val)

    @excepciones(critico=True, imprimir=True)
    def crear(
        self,
        titulo: Optional[Union[str, Sequence[str]]]=None,
        eje_x: Optional[Union[str, Sequence[str]]]=None, eje_y: Optional[Union[str, Sequence[str]]]=None,
        estilo: Sequence[str]=["science", "ieee", "grid"], dpi: int=150
    ) -> tuple[plt.Figure, Union[plt.Axes, list[plt.Axes]]]:
        """
        Crea la figura y los ejes del gráfico, aplicando colores y fuentes predeterminadas.

        Args:
            titulo (str | Sequence[str] | None): Título(s) de los subplots.
            eje_x (str | Sequence[str] | None): Etiqueta(s) del eje X.
            eje_y (str | Sequence[str] | None): Etiqueta(s) del eje Y.
            estilo (Sequence[str]): Estilos de Matplotlib a aplicar.
            dpi (int): Resolución de la figura.

        Returns:
            tuple: (fig, ax) donde `fig` es la figura y `ax` son los ejes creados.
        """
        titulos = self._ajustar_lista(titulo, self.columnas)
        ejes_x = self._ajustar_lista(eje_x, self.columnas)
        ejes_y = self._ajustar_lista(eje_y, self.columnas)

        # Usar atributos de la instancia
        colores = self.colores
        fontsizes = self.fontsizes

        # Crear figura y ejes
        style.use(estilo)
        fig, axes = plt.subplots(1, ncols=self.columnas, figsize=self.tamaño, dpi=dpi)
        fig.patch.set_facecolor(colores["fondo"])
        if self.columnas == 1:              # Forzar a lista
            axes = [axes]
        for i, ax in enumerate(axes):       # Asignar títulos y etiquetas
            if titulos[i]:
                ax.set_title(
                    titulos[i], loc="right", fontsize=fontsizes["titulo"], color=colores["titulo"], fontweight="bold",
                )
            if ejes_x[i]:
                ax.set_xlabel(
                    ejes_x[i],  loc="right", fontsize=fontsizes["eje_x"], color=colores["eje_x"], fontweight="bold", labelpad=2
                )
            if ejes_y[i]:
                ax.set_ylabel(
                    ejes_y[i], loc="top", fontsize=fontsizes["eje_y"], color=colores["eje_y"], fontweight="bold", labelpad=2
                )
            ax.tick_params(
                axis="both", colors=colores["ticks"], labelsize=fontsizes["ticks"]
            )

        self._titulos_asignados = titulos  # Guardar títulos para testing
        
        self.fig, self.ax = fig, axes if self.columnas > 1 else axes[0]
        return self.fig, self.ax

    @excepciones(critico=True, imprimir=True)
    def render(
        self, l_cols: int | list[int]=1, tight_layout: bool=True,
        limite_x: list[tuple[float, float]] | tuple[float, float] | None=None, limite_y: list[tuple[float, float]] | tuple[float, float] | None=None,
        mostrar: bool=True, ruta_guardado: str | None=None
    ) -> tuple[plt.Figure, Union[plt.Axes, list[plt.Axes]]]:
        """
        Configura los ejes de los subplots, muestra y opcionalmente guarda la figura.

        Args:
            l_cols (int | list[int]): Número de columnas de la leyenda por subplot.
            tight_layout (bool): Ajusta automáticamente el layout de la figura.
            limite_x (tuple | list[tuple] | None): Límites de eje X por subplot.
            limite_y (tuple | list[tuple] | None): Límites de eje Y por subplot.
            mostrar (bool): Si True, muestra la figura con plt.show().
            ruta_guardado (str | None): Ruta y nombre del archivo donde guardar la figura. 
                                         Se crea automáticamente la carpeta `Imagenes/`.

        Returns:
            tuple: (fig, ax) con la figura y los ejes configurados.
        """
        fontsizes = self.fontsizes
        # Normalizar argumentos
        if isinstance(l_cols, int):
            l_cols = [l_cols] * self.columnas
        if limite_x is not None and all(isinstance(x, (int, float)) for x in limite_x):
            limite_x =[limite_x] * self.columnas
        if limite_y is not None and all(isinstance(y, (int, float)) for y in limite_y):
            limite_y = [limite_y] * self.columnas

        if isinstance(self.ax, np.ndarray):
            axes_list = self.ax.flatten().tolist()
        elif not isinstance(self.ax, (list, tuple)):
            axes_list = [self.ax]

        # Configuración de cada subplot
        for i, ax in enumerate(axes_list):
            if l_cols[i]:
                ax.legend(
                fontsize=fontsizes["label"],  alignment="left", mode=None,
                frameon=True, framealpha=1, shadow=True,
                ncols=l_cols[i], draggable=True,
                )
            if limite_x is not None and limite_x[i]:
                ax.set_xlim(limite_x[i])
            if limite_y is not None and limite_y[i]:
                ax.set_ylim(limite_y[i])

        # Ajuste de layout
        if tight_layout:
            self.fig.tight_layout()
        # Guardado
        if ruta_guardado is not None:
            archivo = Path("Imagenes") / ruta_guardado
            archivo.parent.mkdir(parents=True, exist_ok=True)   # crea carpeta si hace falta
            self.fig.savefig(archivo, bbox_inches='tight')
            print("Gráfico guardado en:", archivo)
        # Mostrar
        if mostrar:
            plt.show()    
        return self.fig, self.ax