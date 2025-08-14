# Módulo `fit_result` - Contenedor de resultados de ajuste y estadistica posterior

## Descripción general

Este módulo proporciona la clase `FitResult`, diseñada para almacenar y manejar los resultados
de un ajuste estadístico realizado mediante **Orthogonal Distance Regression (ODR)**, utilizando la biblioteca `scipy.odr`.

La clase encapsula los parámetros ajustados con sus incertidumbres, coeficientes de determinación (R² y R² ajustado), residuos, y ofrece métodos auxiliares para análisis de incertidumbres, como el método Jackknife.

La creación de instancias se realiza mediante la función de clase `funciones.fit_odr()` del módulo `Funciones.py` (ver [funciones.md](Funciones.md)).

---

## Dependencias

- `scipy.odr.Output`: Objeto de salida de la rutina ODR.
- `uncertainties.ufloat`: Manejo de valores con incertidumbre.
- `numpy`: Operaciones numéricas y manejo de arrays.
- Decoradores personalizados para control de excepciones (`excepciones`).

---

## Clase `FitResult`

### Atributos principales

| Atributo     | Tipo           | Descripción                                                |
|--------------|----------------|------------------------------------------------------------|
| `ODR_output` | `Output`       | Resultado bruto del ajuste ODR con información completa (objeto `scipy.odr.ODR.run`).   |
| `parametros` | `List[ufloat]` | Parámetros ajustados con valor nominal e incertidumbre (mediante `Uncertainties`).    |
| `R2`         | `Optional[float]` | Coeficiente de determinación (R²) del ajuste.             |
| `R2_aj`      | `Optional[float]` | Coeficiente de determinación ajustado (R² ajustado).      |
| `residuos`   | `Optional[np.ndarray]` | Vector de residuos: diferencia entre datos observados y modelo.|

### Métodos destacados

#### `__str__()`

Devuelve una representación legible y formateada del resultado del ajuste, mostrando parámetros con incertidumbre, valores R² y motivo(s) de finalización del algoritmo ODR.

#### `__iter__()`

Permite iterar sobre los atributos clave del objeto: parámetros, R², R² ajustado, residuos y salida ODR.

#### `jackknife(f, data_x, data_y, ...)`

Calcula la incertidumbre de los parámetros usando el método estadístico **Jackknife**, realizando múltiples ajustes excluyendo datos individuales o subconjuntos según se especifique.

- Parámetros adicionales permiten controlar tolerancias, iteraciones, exclusiones o inclusiones específicas.
- Devuelve una lista de parámetros con incertidumbre estimada y los resultados parciales de cada ajuste.

---

## Ejemplo de uso

```python
from fittools import FR  # from fittools.fit_result import FitResult as fres

# Supongamos que `modelo` es una instancia de Funciones que cuenta con el método fit_odr implementado
resultado = modelo.fit_odr(x_data, y_data, p0=[1.0, 0.5])
print(resultado)

# Estimación de incertidumbres vía Jackknife
parametros_jk, fits_jk = resultado.jackknife(modelo, x_data, y_data)
# Los elementos de fits_jk son objetos creados por fit_odr y por lo tanto son instancias de FitResult en si mimsos
