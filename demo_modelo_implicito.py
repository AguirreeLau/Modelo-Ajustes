"""
Demostracion de flujo completo para ajuste ODR de un modelo implicito (elipse).

Este script:
1. Genera datos sinteticos de una elipse con incertidumbres en x e y.
2. Guarda los datos en CSV.
3. Carga los datos con `DataFrame` y los convierte a `uarray`.
4. Ajusta el modelo implicito con `fit_odr`.
5. Grafica datos + elipse ajustada y guarda la figura.
"""
import numpy as np
import pandas as pd
from uncertainties import unumpy as unp
from pathlib import Path
from fittools import *

# Alias de comodidad para valores nominales y desviaciones estandar
nv, sd = unp.nominal_values, unp.std_devs

# ================================
# 1. Generar datos sinteticos de elipse
# ================================
np.random.seed(7)
h_real, k_real, a_real, b_real = 1.5, -0.8, 4.0, 2.2

t = np.linspace(0, 2 * np.pi, 140, endpoint=False)
err_x = np.full_like(t, 0.04)
err_y = np.full_like(t, 0.04)

x = h_real + a_real * np.cos(t) + np.random.normal(0, err_x, size=t.size)
y = k_real + b_real * np.sin(t) + np.random.normal(0, err_y, size=t.size)

datos_dir = Path("Data")
datos_dir.mkdir(exist_ok=True)
ruta_csv = datos_dir / "demo_elipse_implicito.csv"
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
# 3. Ajuste ODR implicito (elipse)
# ================================
# `funcs.elipse` esta decorada con @modelo_implicito, por lo que `tipo`
# se infiere automaticamente como "implicita".
modelo_elipse = funcs(funcs.elipse)
beta0 = [1.0, -1.0, 3.5, 2.0]  # [h, k, a, b]

resultado = modelo_elipse.fit_odr(
    xdata=nv(X), ydata=nv(Y),
    beta0=beta0,
    errx=sd(X), erry=sd(Y),
    estimadores=True,
)
print(resultado)

# ================================
# 4. Grafico de datos y elipse ajustada
# ================================
graf = grafs(columnas=1)
_, ax = graf.crear(
    titulo="Ajuste de elipse (ODR implicito)",
    eje_x="x",
    eje_y="y",
)

ax.errorbar(
  nv(X), nv(Y), xerr=sd(X), yerr=sd(Y),
  fmt="X", c="#AC1C16", ecolor="#ACACAC",
  label="Datos"
)
x_fit, y_fit = resultado.odrresult.xplusd[0], resultado.odrresult.xplusd[1] 
ax.plot(x_fit, y_fit, label="Elipse ajustada")

graf.render(ruta_guardado="Ajuste_elipse_implicito.png", mostrar=False)