# Modulo `funciones` para ajuste usando `odrpack`

Este modulo define la clase `Funciones`, que encapsula funciones para ajustar datos experimentales usando Orthogonal Distance Regression (ODR) con `odrpack`.
Ademas incluye metodos para calcular estimadores de ajuste y funciones modelo comunes.

---

## Dependencias

Este modulo utiliza las siguientes librerias:

- `numpy`: manejo de arrays y operaciones numericas.
- `odrpack`: ajuste ODR mediante `odr_fit`.
- `uncertainties`: representacion de parametros ajustados con incertidumbre (`valor +- error`).
- `dataclasses`: definicion de clases con atributos.
- `typing`: anotaciones de tipos.
- `_decoradores`: modulo local que provee `@excepciones`.

## Contenidos del modulo

- Clase `Funciones`
  - Atributos principales:
    - `funcion`: funcion callable del modelo.
    - `tipo`: `"explicita"` o `"implicita"` (default: `"explicita"`).
  - Metodos:
    - `__str__`
    - `_check_array`
    - `_peso`
    - `_calcular_estimadores`
    - `_calc_residuos`
    - `_calc_r2`
    - `_calc_r2_ajustado`
    - `_calc_chi2_reducido`
    - `_calc_matriz_correlacion`
    - `fit_odr`
  - Funciones estaticas para modelos comunes:
    - `polinomio`
    - `APV`

---

## Detalle de la clase `Funciones`

### Atributo `funcion`

- Tipo explicito: `Callable[[np.ndarray, np.ndarray], np.ndarray]`.
- Tipo implicito: `Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]`.
- Debe representar el modelo a ajustar.

---

### Metodo `__str__(self)`

- Devuelve nombre y docstring de la funcion contenida.

---

### Metodo `_check_array(self, a, b)`

- Verifica que `a` y `b` tengan igual longitud.
- Lanza `ValueError` si no coinciden.

---

### Estimadores `R2` y `R2 ajustado` (base matematica)

La explicacion matematica del ajuste se mantiene. Sean:

- `{x_i}_{0 <= i <= n}` y `{y_i}_{0 <= i <= n}` muestras de datos relacionadas por una funcion `y = f(x)`.
- `y_i^ = f(x_i)` la prediccion del modelo para la medida i-esima.
- `y_media = (1/n) * sum_{i=0}^n y_i` la media de la variable dependiente.

Se realizan los siguientes calculos:

- **Suma de cuadrados de los residuos (SCR):**
  $$
  \sum_{i=0}^n (y_i - \hat{y}_i)^2
  $$

- **Suma de cuadrados totales (SCT):**
  $$
  \sum_{i=0}^n (y_i - \bar{y})^2
  $$

- **Coeficiente de determinacion `R2`:**
  $$
  R^2 = 1 - \frac{SCR}{SCT}
  $$

- **Coeficiente ajustado `R2_aj`:**
  $$
  R^2_{aj} = 1 - \frac{n - 1}{n - p - 1} (1 - R^2)
  $$
  donde `n` es numero de datos y `p` cantidad de parametros.

En la implementacion actual estos calculos viven en:

- `_calc_r2(sol, xdata, ydata)`
- `_calc_r2_ajustado(sol, xdata, ydata)`

---

### Metodo `fit_odr(self, xdata, ydata, beta0, *, errx=None, erry=None, errx_min=None, erry_min=None, estimadores=None, **kwargs)`

Realiza el ajuste ODR con los siguientes pasos:

1. Valida longitudes de datos y errores con `_check_array`.
2. Valida argumentos de entrada:
   - no permite `errx_min` sin `errx`;
   - no permite `erry_min` sin `erry`;
   - no permite pasar `weight_x`/`weight_y` directo en `kwargs`;
   - exige `beta0` no vacio.
3. Calcula pesos a partir de errores con `_peso`:
   - `weight = 1 / err_eff^2`
   - `err_eff = max(err, err_min)` si hay piso.
4. Ejecuta `odr_fit(self.funcion, xdata, ydata, beta0, weight_x=wx, weight_y=wy, **kwargs)`.
5. Convierte `sol.beta` y `sol.sd_beta` a lista de `ufloat`.
6. Calcula estimadores opcionales con `_calcular_estimadores`.
7. Devuelve `FitResult(odrresult=sol, parametros=..., estimadores=...)`.

**Importante**

- Los parametros optimos se devuelven como `ufloat`.
- Los errores de parametros salen de `sol.sd_beta`, asociados a la covarianza y varianza residual del ajuste.
- El parametro `estimadores` funciona asi:
  - `None`: no calcula estimadores (`{}`).
  - `True`: calcula todos los disponibles.
  - `list/tuple/set`: calcula solo los nombres pedidos.
- Estimadores soportados:
  - `"R2"`
  - `"R2 ajustado"`
  - `"Residuos"`
  - `"Chi2 reducido"`
  - `"Matriz de correlacion"`
- El resultado crudo del ajuste queda en `resultado.odrresult` (tipo `odrpack.OdrResult`).

---

## Funciones estaticas comunes

### `polinomio(x, params)`

Funcion polinomica de grado `n`:

$$
f(x) = a_n x^n + ... + a_1 x + a_0
$$

- No requiere pasar el grado explicitamente.
- Si se usa dentro de `fit_odr`, el grado queda determinado por `len(beta0)-1`.
- Si se usa por fuera de `fit_odr`, el grado queda determinado por `len(params)-1`.

---

### `APV(x, params)`

Modelo **Pseudo-Voigt asimetrico** para ajuste de picos:

- Combina parte Gaussiana y Lorentziana.
- Usa `np.piecewise` para comportamiento distinto en `x <= x0` y `x >= x0`.
- Parametros:
  - `A`: amplitud
  - `x0`: centro del pico
  - `sigma_1`, `eta_1`: ancho y mezcla para `x <= x0`
  - `sigma_2`, `eta_2`: ancho y mezcla para `x >= x0`
  - `y0`: desplazamiento vertical

---

## Ejemplo de uso basico

```python
from fittools import funcs
import numpy as np

func = funcs(funcion=funcs.polinomio)

x_data = np.array([1, 2, 3, 4, 5], dtype=float)
y_data = 2 * x_data + 1 + np.random.normal(0, 0.1, size=x_data.size)

resultado = func.fit_odr(x_data, y_data, beta0=[1, 1], estimadores=True)

print("Parametros ajustados:", resultado.parametros)
print("R2:", resultado.estimadores.get("R2"))
print("Stop reason:", resultado.odrresult.stopreason)
```

---

## Notas finales

- La clase usa `@excepciones` para manejo controlado de errores.
- El ajuste esta basado en `odrpack` (no en `scipy.odr` en la version actual).
- Para detalles de campos avanzados del ajuste, se puede inspeccionar `resultado.odrresult`.
