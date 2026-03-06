# Modulo `fit_result` - Contenedor de resultados de ajuste y analisis posterior

## Descripcion general

Este modulo proporciona la clase `FitResult`, creada para almacenar y manejar resultados de ajustes ODR realizados desde `Funciones.fit_odr`.

En la version actual el ajuste se basa en `odrpack` y no en `scipy.odr`.

La clase encapsula:

- parametros ajustados con incertidumbre,
- resultado crudo de ODR,
- estimadores calculados en un diccionario comun.

La instancia se crea desde `fittools.funciones.Funciones.fit_odr()` (ver [funciones.md](funciones.md)).

---

## Dependencias

- `odrpack.OdrResult`: resultado crudo del ajuste ODR.
- `uncertainties.ufloat`: manejo de valores con incertidumbre.
- `numpy`: operaciones numericas.
- Decoradores personalizados para control de excepciones (`excepciones`).

---

## Clase `FitResult`

### Atributos principales

| Atributo     | Tipo           | Descripcion |
|--------------|----------------|-------------|
| `odrresult`  | `OdrResult`    | Resultado bruto del ajuste ODR (`odrpack`). |
| `parametros` | `List[ufloat]` | Parametros ajustados con valor nominal e incertidumbre. |
| `estimadores`| `dict`         | Diccionario de estimadores calculados (puede venir vacio). |

### Diccionario `estimadores`

Las claves dependen de lo pedido en `fit_odr(..., estimadores=...)` y del tipo
de modelo (`explicita` o `implicita`):

- Modo explicito:
  - `"R2"`
  - `"R2 ajustado"`
  - `"Residuos"`
  - `"Chi2 reducido"`
  - `"Matriz de correlacion"`
- Modo implicito:
  - `"Chi2 reducido"`
  - `"Matriz de correlacion"`
  - `"delta"` (array con forma `(2, n)` como `[delta_x, delta_y]`)
  - `"modulo_delta"` (norma punto a punto de `delta`)

---

## Metodos destacados

### `__str__()`

Devuelve una representacion legible del resultado del ajuste:

- lista de parametros ajustados con incertidumbre,
- `R2 ajustado` (si fue calculado),
- `Chi2 reducido` (si fue calculado),
- motivo de finalizacion (`stopreason`) tomado de `odrresult`.

Si algun estimador no fue calculado, se muestra como `N/A`.

### `__iter__()`

Permite desempaquetar en este orden:

1. `parametros`
2. `estimadores["R2"]`
3. `estimadores["R2 ajustado"]`
4. `estimadores["Residuos"]`
5. `estimadores["Chi2 reducido"]`
6. `estimadores["Matriz de correlacion"]`
7. `odrresult`

Nota: las claves especificas de modelos implicitos (`"delta"` y
`"modulo_delta"`) se consultan directamente desde `resultado.estimadores`.

### `jackknife(f, data_x, data_y, ...)`

Estima incertidumbres de parametros con el metodo Jackknife, realizando multiples ajustes sobre subconjuntos de datos.

- Permite incluir o excluir indices especificos.
- Permite propagar opciones del ajuste (`p0`, tolerancias, iteraciones, etc.).
- Retorna:
  - lista de parametros con incertidumbre Jackknife,
  - lista de ajustes parciales (`fits`), cada uno como `FitResult`.

Errores esperables:

- `ValueError` si se usan simultaneamente `incluir` y `excluir`.
- `TypeError` si `f` no tiene un metodo callable `fit_odr`.
- `RuntimeError` si no se pudo completar ningun ajuste en el proceso.

---

## Ejemplo de uso

```python
from fittools import funcs
import numpy as np

modelo = funcs(funcion=funcs.polinomio)

x_data = np.array([1, 2, 3, 4, 5], dtype=float)
y_data = 2 * x_data + 1

resultado = modelo.fit_odr(x_data, y_data, beta0=[1, 1], estimadores=True)
print(resultado)

print("R2 ajustado:", resultado.estimadores.get("R2 ajustado"))
print("Stop reason:", resultado.odrresult.stopreason)

parametros_jk, fits_jk = resultado.jackknife(modelo, x_data, y_data, p0=[1, 1])
```
