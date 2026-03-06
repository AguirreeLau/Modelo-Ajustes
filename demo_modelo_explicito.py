"""
Demostracion de flujo completo para ajuste ODR de un modelo explicito.

Este script:
1. Genera datos sinteticos de un polinomio cubico con incertidumbres.
2. Guarda los datos en CSV.
3. Carga los datos con `DataFrame` y los convierte a `uarray`.
4. Ajusta el modelo con `fit_odr` en modo explicito.
5. Grafica datos + curva ajustada y guarda la figura.
"""
import numpy as np
import pandas as pd
from uncertainties import unumpy as unp
from pathlib import Path
from fittools import *

# Alias de comodidad para valores nominales y desviaciones estandar
nv, sd = unp.nominal_values, unp.std_devs

# ================================
# 1. Generar datos sinteticos
# ================================
np.random.seed(42)
a, b, c, d = 0.5, -1.2, 3.0, 2.5

x = np.linspace(-5, 5, 60)
err_x = np.full_like(x, 0.05)
err_y = np.full_like(x, 5.0)
ruido = np.random.normal(0, err_y, size=x.size)
y = a * x**3 + b * x**2 + c * x + d + ruido

datos_dir = Path("Data")
datos_dir.mkdir(exist_ok=True)
ruta_csv = datos_dir / "demo_cubica_explicito.csv"
pd.DataFrame(
    {
        "x": x,
        "y": y,
        "error_x": err_x,
        "error_y": err_y,
    }
).to_csv(ruta_csv, index=False)

# ================================
# 2. Cargar datos y convertir a uarray
# ================================
data = dframes.desde_csv(ruta_csv, separacion=",")
xy = data.uarray(str_="error", caso="prefijo")
X, Y = xy["x"], xy["y"]

# ================================
# 3. Ajuste ODR explicito
# ================================
modelo_cubico = funcs(funcs.polinomio)
beta0 = [1.0, -1.0, 1.0, 1.0]

resultado = modelo_cubico.fit_odr(
    xdata=nv(X), ydata=nv(Y),
    beta0=beta0,
    errx=sd(X), erry=sd(Y),
    estimadores=True,
)
print(resultado)

# ================================
# 4. Grafico de datos y ajuste
# ================================
graf = grafs(columnas=1)
_, ax = graf.crear(
    titulo="Ajuste cubico (ODR explicito)",
    eje_x="x",
    eje_y="y",
)

ax.errorbar(
  nv(X), nv(Y), xerr=sd(X), yerr=sd(Y),
  fmt="X", c="#AC1C16", ecolor="#ACACAC",
  label="Datos"
)
x_fit, y_fit = resultado.odrresult.xplusd, resultado.odrresult.yest
ax.plot(x_fit, y_fit, label="Ajuste cubico")

graf.render(ruta_guardado="Ajuste_cubico_explicito.png", mostrar=False)