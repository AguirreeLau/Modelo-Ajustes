# Módulo `Funciones` para ajuste usando scipy.odr

Este módulo define una clase `Funciones` que encapsula funciones para ajustar datos experimentales usando Orthogonal Distance Regression (ODR) con `scipy.odr`. Además incluye métodos para evaluar coeficientes de ajuste y funciones modelo comunes.

---

## Dependencias

Este módulo utiliza las siguientes librerías externas:

- `numpy`: para manejo eficiente de arrays y operaciones numéricas.
- `scipy.odr`: para realizar ajustes por Orthogonal Distance Regression considerando errores en variables independientes y dependientes.
- `uncertainties`: para representar parámetros ajustados con su incertidumbre (valor $\pm$ error).
- `dataclasses`: para definición simplificada de clases con atributos.
- `typing`: para anotaciones de tipos, mejorando la legibilidad y mantenimiento del código.
- `Decoradores`: módulo local que provee el decorador `excepciones` para manejo controlado de errores.

## Contenidos del módulo

- Clase `Funciones`
  - Atributo principal:
    - `funcion`: Función callable que recibe parámetros y array de datos `x`, y devuelve valores `y`.
  - Métodos:
    - `__str__`: devuelve documentación de la función contenida.
    - `_check_array`: verifica que dos arrays tengan igual longitud.
    - `_coef_determinacion`: calcula coeficiente de determinación $\text{R}^2$ y $\text{R}^2$ ajustado.
    - `fit_odr`: realiza ajuste por ODR usando `scipy.odr` con posibilidad de errores en x e y.
  - Funciones estáticas para modelos comunes:
    - `lineal`: modelo lineal $y = m x + b$.
    - `APV`: modelo Pseudo-Voigt asimétrico para ajuste de picos. [BMC Genomics](https://www.biomedcentral.com/epdf/10.1186/1471-2164-16-S12-S12?sharing_token=W6ANcTrMXnAzV6gXiYwDtW_BpE1tBhCbnbw3BuzI2RNFcqE__i7FsnUhh-gOr-2YMGwvRDtSYWobSVIAxt9uCGp4pqiqL8zWY7oz9YwaRSOOek88pW1XdSe8fQpPU5Jye_jeKBSQeGma01hV9zK7lANso1EMt2JwTX7VG4es70k%3D)

---

## Detalle de la clase `Funciones`

### Atributo `funcion`

- Tipo: `Callable[[List[float], np.ndarray], np.ndarray]`
- Debe ser una función que tome:
  - Una lista o array de parámetros libres del modelo.
  - Un array de valores independientes `x`.
- Debe devolver un array con los valores ajustados `y`.

---

### Método `__str__(self)`

- Devuelve el nombre y docstring de la función contenida para facilitar la documentación y comprensión.

---

### Método `_check_array(self, a, b)`

- Verifica que los arrays `a` y `b` tengan igual longitud.
- Retorna `ValueError` si no coinciden.

---

### Método `_coef_determinacion(self, data_y, residuos, cant_params)`

Contiene el cálculo de coeficientes estadísticos que caracterizan la bondad del ajuste. Sean:
- $\{x_i\}_{0 \leq i \leq n}$ y $\{y_i\}_{0 \leq i \leq n}$ muestras de datos que se suponen relacionadas por una función $y=f(x)$.
- $\hat{y}_i = f(x_i)$ la predicción del modelo de la medida i-ésima.
- $\bar{y} = \frac{1}{n} \sum_{i=0}^n y_i$ la media del conjunto de datos que representan la variable dependiente.

se realizan los siguientes cálculos. 
- **Suma de cuadrados de los residuos (SCR):**
  $$
  \sum_{i=0}^n (y_i - \hat{y}_i)^2
  $$

- **Suma de cuadrados totales (SCT):**
  $$
  \sum_{i=0}^n (y_i - \bar{y})^2
  $$

- **Coeficiente de determinación $\text{R}^2$:**
  $$
  \text{R}^2 = 1 - \frac{\text{SCR}}{\text{SCT}}
  $$

- **Coeficiente ajustado $\text{R}^2_{\text{aj}}$:**
  $$
  \text{R}^2_{\text{aj}} = 1 - \frac{n - 1}{n - p - 1} (1 - \text{R}^2)
  $$
  donde $n$ es número de datos y $p$ la cantidad de parámetros.

Retorna una tupla `(R2, R2_aj)`.

---

### Método `fit_odr(self, data_x, data_y, p0, err_x=None, err_y=None, estimadores=True, ...)`

Realiza el ajuste ODR con los siguientes pasos:

1. Valida que los arrays de datos y errores tengan longitudes compatibles.
2. Define el modelo ODR con la función interna.
3. Crea objeto `RealData` con datos y errores (opcional).
4. Ejecuta el ajuste con `ODR`.
5. Extrae parámetros óptimos y sus errores estándar.
6. Calcula coeficientes $\text{R}^2$, $\text{R}^2_{\text{aj}}$ y retorna los residuos si `estimadores=True`.
7. Devuelve un objeto `FitResult` con toda esta información.

**Importante:**

- Los parámetros óptimos se devuelven como un array de `ufloat` (valor $\pm$ error) usando la librería `uncertainties`.
- Los errores en los parámetros son tomados del array `resultado.sd_beta`, calculado como la raiz cuadrada de la diagonal de la matriz covarianza ($\mathbf{C}$) multiplicada la varianza residual estimada ($\mathbf{s}^2$).
  $$ \mathbf{C} = 
  \left(
  \begin{matrix}
  \sigma_{\beta_1 \beta_1} & \sigma_{\beta_1 \beta_2} & \cdots & \sigma_{\beta_1 \beta_n} \\
  \sigma_{\beta_2 \beta_1} & \sigma_{\beta_2 \beta_2} & \cdots & \sigma_{\beta_2 \beta_n} \\
  \vdots & \vdots & \ddots & \vdots \\
  \sigma_{\beta_n \beta_1} & \sigma_{\beta_n \beta_2} & \cdots & \sigma_{\beta_n \beta_n}
  \end{matrix}
  \right)
  =
  \left(
  \begin{matrix}
  \sigma_{\beta_1}^2 & \sigma_{\beta_1 \beta_2} & \cdots & \sigma_{\beta_1 \beta_n} \\
  \sigma_{\beta_2 \beta_1} & \sigma_{\beta_2}^2 & \cdots & \sigma_{\beta_2 \beta_n} \\
  \vdots & \vdots & \ddots & \vdots \\
  \sigma_{\beta_n \beta_1} & \sigma_{\beta_n \beta_2} & \cdots & \sigma_{\beta_n}^2
  \end{matrix}
  \right)
  $$
  donde $\{\beta_i\}_{0 \leq i \leq n}$ son los parámetros óptimos ajustados. Luego, la desviación estandar de cada parámtero es
  $$
  \mathbf{sd}_\beta = \sqrt{\text{diag}(\mathbf{C})\cdot \mathbf{s}^2} =
  \left(
  \begin{matrix}
  \sigma_{\beta_1} \\
  \sigma_{\beta_2} \\
  \vdots \\
  \sigma_{\beta_n}
  \end{matrix}
  \right),
  \quad \text{donde} \quad
  \mathbf{s}^2 = \frac{\text{SCR}}{n-p}
  $$

- Se pueden controlar opciones de tolerancia y cantidad máxima de iteraciones del ODR. Los argumentos `sstol` y `partol` controlan la tolerancia del ajuste para determinar cuando se estabilizó la convergencia de la suma de cuadrados (en el primer caso) o de los parámetros (segundo caso). Por otro lado `maxit` establece un límite máximo para las iteraciones que realiza el ajuste.

- Se puede acceder a los objetos originales creados por scipy.odr luego del ajuste, esto es `resultado.ODR_output` (ver [FitResult.md](FitResult.md)) para obtener atributos no calculados por fit_odr como por ejemplo `cov_beta`. [Lista de atributos](https://docs.scipy.org/doc/scipy/reference/generated/scipy.odr.Output.html#scipy.odr.Output) - [Otra fuente](https://tedboy.github.io/scipy/generated/scipy.odr.Output.html#scipy.odr.Output)

---

## Funciones estáticas comunes

### `polinomio(params, x)`

Función polinómica de grado $n$
$$
f(x) = a_nx^n + ... + a_1x + a_0
$$

- No necesita que se pase el grado como argumento, hereda el grado que se le pasa al establecer los parámetros iniciales de busqueda en `fit_odr` con `p0`.
- Dado `p0`, el polinomio es de grado `len(p0)-1`.
- De llamarse por fuera de la función `fit_odr` se especifica el grado con la lista `params` que se pase.
- Parámetros:
  - `a_0`: Término independiente.
  - `a_1`: Coeficiente del término lineal.
  - ...
  - `a_n`: Coeficiente del término de mayor grado.

---

### `APV(params, x)`

Modelo **Pseudo-Voigt asimétrico** para ajuste de picos:

- Combina función Gaussiana y Lorentziana con parámetros que controlan amplitud, posición, anchos y asimetría.
- Usa `np.piecewise` para modelar comportamiento distinto para $x < x_0$ y $x \geq x_0$.
- Parámetros:
  - `A`: amplitud
  - `x0`: centro del pico
  - `sigma_1`, `eta_1`: ancho y mezcla Gaussiana/Lorentziana para $x < x_0$
  - `sigma_2`, `eta_2`: ancho y mezcla para $x \geq x_0$
  - `y0`: desplazamiento vertical

---


## Ejemplo de uso básico

```python
from Clases import F  # from Clases.Funciones import Funciones as F 
import numpy as np

# Crear instancia tomando la función polinómica (de grado 1) de la clase
func = Funciones(funcion=F.polinomio)

# Datos simulados
x_data = np.array([1,2,3,4,5])
y_data = 2 * x_data + 1 + np.random.normal(0, 0.1, size=x_data.size)

# Ajustar
resultado = func.fit_odr(x_data, y_data, p0=[1,1])

print("Parámetros ajustados:", resultado.parametros)
print("R2:", resultado.R2)
```
---

## Notas finales

- La clase utiliza un decorador `@excepciones` para manejo controlado de errores.
- El módulo integra librerías para manejo de errores en datos y parámetros (`uncertainties`).
- El módulo está principalmente basado en [scipy.odr](https://docs.scipy.org/doc/scipy/reference/odr.html) y pretende ser una colección útil de sus funciones para trabajo habitual de laboratorio. De ser necesario, se permite el acceso a los objetos creados por el proyecto original para profundizar el análisis.