"""
Módulo Data para manejo avanzado de datos tabulares.

Este módulo contiene la clase `DataFrame`, que permite cargar, filtrar
y manipular datos tabulares en pandas con funcionalidades adicionales,
como manejo de incertidumbres y filtrado flexible.

## Funciones principales
    - Carga de archivos CSV con control de encabezados y filas a ignorar.
    - Filtrado avanzado con múltiples formatos de condiciones.
    - Conversión a arrays con incertidumbre usando `uncertainties.unumpy`.
    - Generación de grillas para visualización y cálculo de gradientes.

## Dependencias
    - pandas
    - numpy
    - uncertainties
    - re (expresiones regulares)
    - os
"""
from ._decoradores import excepciones
from typing import Union, Optional, Tuple, Dict
from dataclasses import dataclass
from uncertainties import unumpy as unp
import pandas as pd
import numpy as np
import re
import os

@dataclass
class DataFrame:
    """
    Clase para cargar y manipular datos tabulares de forma flexible.

    ## Atributos
        - path (str): Ruta al archivo fuente de datos.
        - df (Optional[pd.DataFrame]): DataFrame interno que contiene los datos cargados.

    ## Métodos destacados
        - desde_csv: Constructor alternativo para crear instancia desde archivo CSV.
        - filtrar: Filtra el DataFrame con múltiples formas de especificar la condición.
        - uarray: Convierte columnas con errores asociados a arrays con incertidumbre.
        - grilla_y_grad: Genera grilla 2D y calcula gradientes para visualización.

    ## Uso típico
        ```python
        df = DataFrame.desde_csv("datos.csv")
        df_filtrado = df.filtrar("col1 > 0")
        ```
    """
    path: str
    df: Optional[pd.DataFrame] = None

    def __str__(self) -> str:
        """Representación legible del DataFrame. Igual a
        ```python 
        print(instancia.df)
        ```"""
        if self.df is None:
            return "DataFrame vacío. No se han cargado datos."
        return f"{self.df}"

    @excepciones(critico=True, imprimir=True)
    def _leer_datos(self, separacion: str = "\t", encabezados: Optional[bool] = True, nombres: Optional[list[str]]= None, ignorar: Optional[Union[int, list[int]]] = None) -> None:
        """
        Método interno que carga los datos desde el archivo especificado en ``self.path``.

        Este método encapsula la lógica de lectura del archivo y la carga en el DataFrame interno.
        No debería ser invocado directamente desde fuera de la clase, salvo por métodos públicos como ``desde_csv``.

        Args:
            separacion (str): Caracter separador entre columnas en el archivo. Por defecto es tabulación.
            encabezados (bool, opcional): Indica si la primera fila contiene los nombres de las columnas.
                                        Si es False, se deben proporcionar nombres explícitos mediante ``nombres``.
            nombres (list de str, opcional): Lista con nombres de columnas a usar si ``encabezados`` es False.
            ignorar (int o lista de int, opcional): Filas a omitir al leer el archivo. Puede ser un entero (número de filas iniciales) o una lista de índices.

        Raises:
            ValueError: Si ``encabezados`` es False y ``nombres`` es None.
            TypeError: Si ``nombres`` no es una lista de strings válida.

        Examples:
            ```python
            self._leer_datos(separacion=',', encabezados=False, nombres=['A', 'B', 'C'], ignorar=[0, 2])
            ```
        """
        if not encabezados:
            if nombres is None:
                raise ValueError("Se debe proporcionar una lista de nombres si encabezados=False.")
            if not isinstance(nombres, list) or not all(isinstance(n, str) for n in nombres):
                raise TypeError("El parámetro 'nombres' debe ser una lista de strings.")

        self.df = pd.read_csv(
            self.path,
            sep=separacion, on_bad_lines="warn",
            header=0 if encabezados else None,
            names=nombres if not encabezados else None,
            skiprows=ignorar
        )
        print(f"Archivo leido correctamente desde {self.path}")     

    @classmethod
    @excepciones(critico=True, imprimir=True)
    def desde_csv(cls, path, separacion: str = "\t",  encabezados: Optional[bool] = True, nombres: Optional[list[str]]= None, ignorar: Optional[Union[int, list]] = None) -> "DataFrame":
        """
        Crea una instancia de ``DataFrame`` leyendo datos desde un archivo CSV.

        Este método es un constructor alternativo que carga automáticamente los datos
        desde el archivo ubicado en ``path``, aplicando las opciones de separación,
        encabezados y filas a ignorar.

        Args:
            path (str): Ruta al archivo CSV.
            separacion (str): Separador de campos en el archivo. Por defecto es tabulación.
            encabezados (bool, opcional): Indica si la primera fila contiene nombres de columnas. Si es False, se deben proporcionar nombres.
            nombres (list, opcional): Lista de nombres de columnas a usar si ``encabezados`` es False.
            ignorar (int o list, opcional): Número(s) de línea(s) a ignorar al leer el archivo. Compatible con el argumento ``skiprows`` de pandas.

        Returns:
            DataFrame: Instancia de la clase con los datos ya cargados.

        Raises:
            FileNotFoundError: Si la ruta al archivo no existe.
            ValueError/TypeError: Si los parámetros son inconsistentes o inválidos.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"El archivo {path} no existe.")

        instancia = cls(path=path)
        instancia._leer_datos(separacion, encabezados, nombres, ignorar)
        return instancia

    @excepciones(critico=True, imprimir=True)
    def filtrar(self, condicion: Union[str, pd.Series, callable], preservar_no_especificados : bool = True) -> "DataFrame":
        """
        Filtra el DataFrame interno y devuelve una nueva instancia con el subconjunto seleccionado.
        Se aceptan cuatro formas de especificar la condición:
    
            Un string estilo query de pandas, por ejemplo
            ```python
            col1 > 0 and col2 == 'A'
            ```

            Una lista o tupla de strings, que se aplican en cascada como intersección (AND): cada expresión se pasa
                sucesivamente a "query" sobre el resultado previo.

            Una "pd.Series" booleana: debe tener dtype booleano; si su índice difiere del del DataFrame original, se
                reindexa. Los índices no especificados se rellenan con "True" o "False" según ``preservar_no_especificados``.

            Un "callable" que recibe el DataFrame y retorna una ``pd.Series`` booleana con el mismo índice (o que será
                reindexada según ``preservar_no_especificados``).

        Args:
            condicion (str | list[str] | tuple[str, ...] | pd.Series | callable): Criterio de filtrado.
            preservar_no_especificados (bool): Si la máscara booleana (de Serie o callable) tiene índice distinto, los valores faltantes se completan con "True" cuando es True (incluir) o con "False" cuando es False (excluir).

        Returns:
            DataFrame (DataFrame): Nueva instancia de la misma clase que contiene el DataFrame filtrado.

        Raises:
            ValueError: Si no hay DataFrame cargado, si la condición es de tipo inválido, si una Serie no es booleana, si el callable no retorna una Serie booleana, o si alguna expresión de query falla.

        Examples:
            Usando un string único
            ```python
            df_filtrado = instancia.filtrar("col1 > 0 and col2 == 'A'")
            ```

            Varias condiciones en lista (intersección)
            ```python
            df_filtrado = instancia.filtrar(["col1 > 0", "col2 == 'A'"])
            ```

            Serie booleana con índice distinto, rellenando no especificados con False
            ```python
            mask = pd.Series([True, False], index=[0, 2])
            df_filtrado = instancia.filtrar(mask, preservar_no_especificados=False)
            ```

            Callable complejo
            ```python
            df_filtrado = instancia.filtrar(lambda df: (df["x"] > 5) & (df["y"].abs() < 10))
            ```

        ## Notes
            Para combinar condiciones con OR en vez de AND hay que construir la Serie o string adecuado fuera de esta función.
            El reindexado de máscaras con índice distinto usa el comportamiento controlado por `preservar_no_especificados` para evitar introducir NaNs que rompan la indexación.
        """
        if self.df is None:
            raise ValueError("DataFrame no ha sido cargado.")

        elif isinstance(condicion, str) or isinstance(condicion, (list, tuple)) and all(isinstance(c, str) for c in condicion):
            condicion = [condicion] if isinstance(condicion, str) else condicion
            df_filtrado = self.df
            try:
                for cond in condicion:
                    df_filtrado = df_filtrado.query(cond)
            except Exception as e:
                raise ValueError(f"Error al evaluar una condición en la lista: {e}")
        
        elif isinstance(condicion, pd.Series):
            if condicion.dtype != bool:
                raise ValueError("La Series pasada como condición debe ser booleana.")
            if not condicion.index.equals(self.df.index):
                if preservar_no_especificados == True:
                    condicion = condicion.reindex(self.df.index, fill_value=True)
                else :
                    condicion = condicion.reindex(self.df.index, fill_value=False)
            df_filtrado = self.df.loc[condicion]
        
        elif callable(condicion):
            try:
                máscara = condicion(self.df)
                if not isinstance(máscara, pd.Series) or máscara.dtype != bool:
                    raise ValueError("El callable debe devolver una pd.Series booleana.")
                if not máscara.index.equals(self.df.index):
                    if preservar_no_especificados == True:
                        máscara = máscara.reindex(self.df.index, fill_value=True)
                    else :
                        máscara = máscara.reindex(self.df.index, fill_value=False)
                df_filtrado = self.df.loc[máscara]
            except Exception as e:
                raise ValueError(f"Error al ejecutar el callable de filtrado: {e}")
        
        else:
            raise ValueError("Condición de filtrado inválida. Debe ser str, pd.Series booleana o callable.")

        nueva = type(self)(path=self.path, df=df_filtrado)
        return nueva

    @excepciones (critico=True, imprimir=True)                
    def separar_c_f(self, caso: str) -> list[np.ndarray]:
        """
        Devuelve una lista de arrays con los datos del DataFrame separados por filas o columnas.

        Args:
            caso (str): Indica la separación deseada:
                        - "f": separa por filas.
                        - "c": separa por columnas.

        Returns:
            list[np.ndarray]: Lista de arrays numpy correspondientes a cada fila o columna,
                            con elementos numéricos convertidos a float cuando sea posible.

        Raises:
            ValueError: Si el DataFrame no ha sido cargado o si `caso` no es "f" ni "c".

        Example:
            ```python
            filas = instancia.separar_c_f("f")
            columnas = instancia.separar_c_f("c")
            ```
        """
        if self.df is None:
            raise ValueError("DataFrame no ha sido cargado.")
        if caso not in {"f", "c"}: 
            raise ValueError("El caso debe ser 'f' para filas o 'c' para columnas.")

        if caso == "f":     # Itera sobre cada fila del DataFrame (como arrays 1D)
            return [
                np.array([
                    float(v) if self._es_numerico(v) else v     # Convierte a float si es posible
                    for v in fila                               # Recorre cada valor de la fila
                ]) for fila in self.df.values                   # Obtiene las filas como arrays desde el DataFrame
            ]
        else:       # Itera sobre cada columna del DataFrame (como arrays 1D, usando la transpuesta)
            return [
                np.array([
                    float(v) if self._es_numerico(v) else v     # Convierte a float si es posible
                    for v in col                                # Recorre cada valor de la columna
                ]) for col in self.df.T.values                  # Obtiene las columnas transponiendo el DataFrame
            ]

    @staticmethod
    @excepciones(critico=True, imprimir=True)
    def _norm_str(s: str) -> str:
        """
        Método interno. Normaliza un string para facilitar comparaciones entre nombres de columnas.

        ## Cambios
            Convierte a minúsculas.
            Elimina espacios al principio y final.
            Reemplaza espacios, guiones y puntos por guion bajo.
            Elimina caracteres no alfanuméricos o guion bajo.
            Elimina guiones bajos repetidos (más de uno) por uno solo

        ## Examples
            ```python
            s = "Temperatura (°C)"
            s_normalizado = DataFrame._norm_str(s)
            print(s_normalizado)  # "temperatura_c"
            ```
        """
        s = s.strip().lower()
        s = re.sub(r"[ \-\.]+", "_", s)     # Reemplaza secuencias de espacios, guiones o puntos por un solo "_"
        s = re.sub(r"_+", "_", s)           # Elimina guiones bajos repetidos (más de uno) por uno solo
        s = s.strip("_")                    # Elimina guion bajo al principio o final
        s = re.sub(r"[^a-z0-9_]", "", s)    # Elimina caracteres que no sean letras, números o guion bajo
        return s

    @excepciones (critico=True, imprimir=True)  
    def uarray(self, str_: str = "err", caso: str = "sufijo") -> Dict[str, np.ndarray]:
        """
        Convierte columnas nominales y sus columnas de error asociadas en arrays de números con incertidumbre.

        Para cada columna nominal en el DataFrame, busca la columna de error correspondiente según un
        identificador (`str_`) como sufijo o prefijo (definido por `caso`), y construye arrays de
        valores con incertidumbres usando `uncertainties.unumpy.uarray`.

        Args:
            `str_` (str): Cadena que identifica la columna de error (por ejemplo, "err" o "sigma").
            caso (str): Indica si el identificador está como "sufijo" o "prefijo" en el nombre de la columna.

        Returns:
            Dict(str, np.ndarray): Diccionario donde cada clave es el nombre de una columna nominal y el valor
                                    es un array ``unumpy.uarray`` con valores e incertidumbres.

        Raises:
            ValueError: Si el DataFrame no ha sido leído o si 'caso' no es 'sufijo' o 'prefijo'.

        Example:
        ```python
        arrays = instancia.uarray(str_='err', caso='sufijo')
        ```
        """
        if self.df is None:
            raise ValueError("DataFrame no ha sido leído aún.")
        if caso not in {"sufijo", "prefijo"}: 
            raise ValueError("El caso debe ser 'sufijo' o 'prefijo'.")

        unumpy_arrays = {}
        cols_original = list(self.df.columns)
        cols_norm = [self._norm_str(c) for c in cols_original]
        norm_to_orig = dict(zip(cols_norm, cols_original))      # Mapea nombres normalizados a originales

        for col_orig, col_norm in zip(cols_original, cols_norm):
            if caso == "sufijo":
                target_norm = f"{col_norm}_{str_}"
                if target_norm in norm_to_orig:
                    err_col_orig = norm_to_orig[target_norm]
                    valores = self.df[col_orig].to_numpy(dtype=float)
                    errores = self.df[err_col_orig].to_numpy(dtype=float)
                    unumpy_arrays[col_orig] = unp.uarray(valores, errores)

            elif caso == "prefijo":
                target_norm = f"{str_}_{col_norm}"
                if target_norm in norm_to_orig:
                    err_col_orig = norm_to_orig[target_norm]
                    valores = self.df[col_orig].to_numpy(dtype=float)
                    errores = self.df[err_col_orig].to_numpy(dtype=float)
                    unumpy_arrays[col_orig] = unp.uarray(valores, errores)
        return unumpy_arrays

    @excepciones(critico=True, imprimir=True)
    def grilla_y_grad(self, centrar: bool = True, grad: bool = True) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Genera una grilla 2D para visualizar el DataFrame como superficie.

        Args:
            centrar (bool): Si es True, la grilla se centra en el origen con coordenadas simétricas alrededor de 0.
                            Para dimensiones impares, el centro coincide con un punto; para pares, queda entre dos.
                            Si es False, la grilla va desde (0, 0) hasta (n-1, m-1).
            grad (bool): Si es True, se calcula el gradiente en X e Y del DataFrame; si es False, no se calcula.

        Returns:
            Tuple [np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
                - X (np.ndarray): Coordenadas en X de la grilla (columnas).
                - Y (np.ndarray): Coordenadas en Y de la grilla (filas).
                - dx (np.ndarray o None): Gradiente en X si `grad` es True; None en caso contrario.
                - dy (np.ndarray o None): Gradiente en Y si `grad` es True; None en caso contrario.


        Raises:
            ValueError: Si el DataFrame no ha sido cargado.

        Example:
             ```python
            X, Y, dx, dy = instancia.grilla_y_grad(centrar=True, grad=True)
            ```
        """
        if self.df is None:
            raise ValueError("DataFrame no ha sido cargado.")

        dx, dy = None, None
        n_f, n_c = self.df.shape
        if centrar:
            rango_f = np.linspace(-(n_f - 1) / 2, (n_f - 1) / 2, n_f)
            rango_c = np.linspace(-(n_c - 1) / 2, (n_c - 1) / 2, n_c)
        else:
            rango_f = np.linspace(0, n_f - 1, n_f)
            rango_c = np.linspace(0, n_c - 1, n_c)
        X, Y = np.meshgrid(rango_c, rango_f)    # X: columnas, Y: filas

        if grad:
            dx, dy = np.gradient(self.df.values)
        return X, Y, dx, dy

    @staticmethod
    def _es_numerico(value) -> bool:
        """
        Método interno. Verifica si un valor es numérico (int o float).

        Args:
            value: Valor a evaluar.

        Returns:
            bool: True si ``value`` puede convertirse a float, False en caso contrario.

        Example:
            ```python
            es_num = DataFrame._es_numerico("3.14")  # True
            es_num = DataFrame._es_numerico("abc")   # False
            ```
        """
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False