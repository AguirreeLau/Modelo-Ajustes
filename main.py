"""
Demostración de flujo completo del proyecto de ajuste cúbico con incertidumbres.

Este script realiza las siguientes operaciones:

### 1. Generación de datos sintéticos
   - Se crea una función cúbica `y = a*x^3 + b*x^2 + c*x + d`.
   - Se agregan incertidumbres en x (`ux`) y en y (`uy`) y ruido gaussiano.
   - Los datos se guardan en un CSV en la carpeta `datos/`.

### 2. Carga y transformación de datos
   - Se utiliza la clase `DF` para leer el CSV.
   - Se convierte cada columna nominal y su columna de error asociada
     en arrays de números con incertidumbre usando `uarray` de `uncertainties`.

### 3. Ajuste de los datos con ODR
   - Se define la función cúbica con `F.polinomio`.
   - Se realiza el ajuste usando `fit_odr` considerando incertidumbres
     en x e y.
   - El resultado es un objeto `FitResult` que contiene parámetros ajustados,
     incertidumbres, R² y residuos.

### 4. Visualización
   - Se crea un gráfico con `G`.
   - Se plotean los datos con barras de error.
   - Se plotea la curva ajustada.
   - Se guarda la figura resultante como `"Ajuste cúbico.png"`.

### Dependencias
    - numpy, pandas
    - uncertainties
    - matplotlib
    - Base (módulos propios: Data, Funciones, Graficos, FitResult)
"""
from uncertainties import unumpy as unp
# Renombramos funciones de unumpy por comodidad
nv = unp.nominal_values
sd = unp.std_devs

import numpy as np
import pandas as pd
from pathlib import Path
from Base import *

# ================================
# 1. Generar datos
# ================================
np.random.seed(42)  # reproducible
a, b, c, d = 0.5, -1.2, 3.0, 2.5

x = np.linspace(-5, 5, 50)
ux = np.full_like(x, 0.05)  # incertidumbre fija en x

# Ruido y en base a dispersión + incertidumbre fija
ruido = np.random.normal(0, 5, size=len(x))
y = a*x**3 + b*x**2 + c*x + d + ruido
uy = np.full_like(y, 5.0)  # incertidumbre fija en y

# Guardar en carpeta datos
datos_dir = Path("datos")
datos_dir.mkdir(exist_ok=True)
df_demo = pd.DataFrame({"x": x, "y": y, "error x": ux, "error y": uy})
df_demo.to_csv(datos_dir / "demo_cubica.csv", index=False)

# ================================
# 2. Cargar datos con Data
# ================================
data = DF.desde_csv(datos_dir / "demo_cubica.csv", separacion=",")
XY = data.uarray(str_="error", caso="prefijo")
X, Y = XY["x"], XY["y"]

# ================================
# 4. Ajuste usando ODR con incertidumbre
# ================================
func_cubica = F(F.polinomio)
p0 = [1, -1, 1, 1]  # valores iniciales
resultado = func_cubica.fit_odr(
    data_x = nv(X), data_y = nv(Y), p0 = p0,
    err_x = sd(X), err_y = sd(Y)
)
print(resultado)

# ================================
# 5. Graficar datos y ajuste
# ================================
graf = G(columnas=1)
fig, ax = graf.crear(titulo="Ajuste cúbico", eje_x="x", eje_y="y")

# Graficar datos con incertezas
ax.errorbar(
    nv(X), nv(Y), xerr=sd(X), yerr=sd(Y),
    fmt="X", c="#AC1C16", ecolor="#ACACAC",
    label="Datos"
)
# Graficar ajuste cúbico
y_fit = F.polinomio(nv(resultado.parametros), nv(X))
ax.plot(nv(X), y_fit, label="Ajuste cúbico")

graf.render(ruta_guardado="Ajuste_cubico.png", mostrar=False)