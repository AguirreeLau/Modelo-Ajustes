# Módulo `data_frames` — Manejo avanzado de datos tabulares

Este módulo provee la clase `DataFrame`, diseñada para cargar, filtrar y manipular datos tabulares con funcionalidades extendidas que complementan a `pandas.DataFrame`, incluyendo manejo de incertidumbres y filtrado flexible.

---

## Dependencias

- `pandas`: manejo y manipulación de datos tabulares.
- `numpy`: operaciones numéricas y manejo de arrays.
- `uncertainties` (`unumpy`): creación y manipulación de arrays con incertidumbres.
- `re`: expresiones regulares para procesamiento de strings.
- `os`: operaciones del sistema de archivos.
- `_decoradores`: Decoradores personalizados para manejo de excepciones (`excepciones`).

---

## Clase `DataFrame`

Clase principal para trabajar con datos tabulares, especialmente útil para datos experimentales con incertidumbre.

### Atributos

| Nombre | Tipo | Descripción |
|--------|------|-------------|
| `path` | `str` | Ruta al archivo de datos fuente (csv). |
| `df` | `Optional[pd.DataFrame]` | DataFrame interno que almacena los datos cargados. Puede ser `None` si no se cargó ningún dato. |

---

### Métodos principales

#### `desde_csv`

Constructor alternativo para crear una instancia a partir de un archivo CSV.

- Parámetros:  
  - `path (str)`: ruta al archivo.  
  - `separacion (str, default="\t")`: separador de columnas en el archivo.  
  - `encabezados (bool, default=True)`: si el archivo tiene fila de encabezados.  
  - `nombres (list[str], opcional)`: nombres de columnas si no hay encabezados.  
  - `ignorar (int o lista[int], opcional)`: filas a omitir al leer.

- Retorna:  
  - Una instancia `DataFrame` con datos cargados.

- Errores:  
  - `FileNotFoundError` si no existe el archivo.  
  - `ValueError` o `TypeError` si parámetros inconsistentes.

---

#### `filtrar`

Filtra los datos aplicando condiciones flexibles.

- Condiciones aceptadas:  
  - String estilo pandas `query` (ej: `"col1 > 0 and col2 == 'A'"`).  
  - Lista/tupla de strings, se aplican en cascada (AND).  
  - `pd.Series` booleana, con reindexado controlado.  
  - Callable que recibe el DataFrame y retorna `pd.Series` booleana.

- Parámetros:  
  - `condicion`: condición para filtrar (tipos mencionados).  
  - `preservar_no_especificados` (bool): cómo rellenar índices faltantes al reindexar (True: incluir, False: excluir).

- Retorna:  
  - Nueva instancia `DataFrame` con subconjunto filtrado.

- Errores:  
  - `ValueError` si condición inválida o error al evaluarla.

---

#### `uarray`

Genera arrays con incertidumbre combinando columnas nominales y sus errores asociados.

- Parámetros:  
  - `str_` (str): sufijo o prefijo identificador de columna de error (ej: `"err"`).  
  - `caso` (str): `"sufijo"` o `"prefijo"` para determinar posición del identificador.

- Retorna:  
  - Diccionario con clave = columna nominal, valor = array con incertidumbre (`uncertainties.unumpy.uarray`).

- Errores:  
  - `ValueError` si el DataFrame no está cargado o `caso` inválido.

---

#### `grilla_y_grad`

Genera una grilla 2D para visualización y calcula gradientes si se solicita.

- Parámetros:  
  - `centrar (bool, default=True)`: si la grilla se centra en cero.  
  - `grad (bool, default=True)`: si calcula gradientes en X e Y.

- Retorna:  
  - Tupla: `(X, Y, dx, dy)` donde `X`, `Y` son coordenadas de la grilla y `dx`, `dy` gradientes (o `None` si no se calculan).

- Errores:  
  - `ValueError` si DataFrame no cargado.

---

### Métodos auxiliares

- `_leer_datos`: método interno para cargar datos con opciones avanzadas.  
- `_norm_str`: normaliza nombres de columnas para buscar sufijos o prefijos de errores.  
- `_es_numerico`: chequea si un valor es convertible a float.  
- `separar_c_f`: devuelve listas de arrays separando filas o columnas.

---

## Uso típico

```python
from fittools import dframes  # from fittools.data_frames import DataFrame as dframes

# Cargar datos desde CSV
df = dframes.desde_csv("datos.csv", separacion=",", encabezados=True)

# Filtrar filas donde 'temperatura' > 20 y 'presion' < 1.5
df_filtrado = df.filtrar("temperatura > 20 and presion < 1.5")

# Obtener arrays con incertidumbre de columnas nominales y sus errores asociados con sufijo '_err'
arrays_con_inc = df.uarray(str_="err", caso="sufijo")

# Generar grilla centrada y gradiente para visualización
X, Y, dx, dy = df.grilla_y_grad(centrar=True, grad=True)
