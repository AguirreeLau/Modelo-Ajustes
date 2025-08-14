# Módulo `Graficos` - Estandarización de gráficos por Matplotlib

El módulo `Graficos` proporciona una clase para crear, configurar, mostrar y guardar gráficos de manera flexible utilizando Matplotlib, con soporte para múltiples subplots, estilos científicos, colores y tamaños de fuente personalizables.

---

## Dependencias

- `matplotlib` para la creación de gráficos.
- `scienceplots` para estilos científicos adicionales.
- `numpy` para manejo de arrays.
- `pathlib` para manejo de rutas de archivos.
- `Data` para la función `_norm_str` (normalización de strings).
- `Decoradores` para el decorador `excepciones`.

---

## Clase `Graficos`

### Descripción

`Graficos` es una clase diseñada para simplificar la creación de gráficos en Python, permitiendo:

- Configurar subplots múltiples.
- Aplicar estilos científicos (`scienceplots`).
- Personalizar colores y tamaños de fuente para títulos, ejes, ticks, leyendas y fondo.
- Guardar automáticamente los gráficos en carpetas.
- Normalizar la entrada de diccionarios y listas de manera flexible.

### Atributos

| Atributo      | Tipo           | Descripción |
|---------------|----------------|-------------|
| `columnas`    | `int`          | Número de columnas de subplots (default: 1) |
| `tamaño`      | `tuple[int,int]` | Tamaño de la figura en pulgadas `(ancho, alto)` |
| `colores`     | `dict[str,str]` | Colores para títulos, ejes, ticks, grilla y fondo |
| `fontsizes`   | `dict[str,int]` | Tamaños de fuente para títulos, ejes, ticks y label |

---

### Métodos

#### `crear()`

Crea la figura y los ejes del gráfico.

**Descripción detallada:**

- Ajusta automáticamente la longitud de las listas de títulos y etiquetas si se pasan valores individuales.
- Aplica estilos con `scienceplots`, aunque acepta cualquiera con soporte de matplotlib.
- Asigna colores y tamaños de fuente desde los atributos de la instancia (`colores` y `fontsizes`).
- Configura la figura y los ejes según la cantidad de columnas.
- Devuelve los objetos `fig` y `ax` para que puedan usarse posteriormente para graficar datos.

**Parámetros:**

- `titulo` (`str` o `Sequence[str]`, opcional): Título(s) de los subplots.
- `eje_x` (`str` o `Sequence[str]`, opcional): Etiqueta(s) del eje X.
- `eje_y` (`str` o `Sequence[str]`, opcional): Etiqueta(s) del eje Y.
- `estilo` (`Sequence[str]`): Estilos de Matplotlib a aplicar (default `["science","ieee","grid"]`).
- `dpi` (`int`): Resolución de la figura.

**Retorna:**

- `fig` (`plt.Figure`): Figura creada.
- `ax` (`plt.Axes` o lista de `plt.Axes`): Ejes de los subplots.

#### `render()`

Configura los ejes, muestra y guarda la figura.

**Descripción detallada:**

- Ajusta la cantidad de columnas de la leyenda si se pasa un solo valor o lista.
- Aplica límites a los ejes X e Y si se proporcionan.
- Configura leyendas en cada subplot con estilos y tamaños de fuente definidos en la instancia.
- Ajusta automáticamente el layout si `tight_layout=True`.
- Guarda la figura en la ruta especificada creando carpetas necesarias.
- Muestra la figura con `plt.show()` si `mostrar=True`.
- Guarda el gráfico en una carpeta Imagenes y la crea en el caso de que no exista.
- Devuelve la figura y los ejes para uso posterior.

**Parámetros:**

- `l_cols`: `int` o `list[int]`, columnas de leyenda.
- `tight_layout`: `bool`, ajuste automático del layout.
- `limite_x`: `tuple` o `list[tuple]`, límites eje X.
- `limite_y`: `tuple` o `list[tuple]`, límites eje Y.
- `mostrar`: `bool`, si se muestra la figura.
- `ruta_guardado`: `str` o `None`, ruta para guardar la figura.

**Retorna:**

- `fig`: `plt.Figure`
- `ax`: `plt.Axes` o lista de `plt.Axes`

---

## Ejemplo de uso
```python
from Clases import G  # from Clases.Graficos import Graficos as G
import numpy as np

g = Graficos(columnas=2)    # Crear instancia

# Crear figura y ejes
fig, ax = g.crear(titulo=["Temperatura","Presión"], eje_x="Tiempo", eje_y=["Temp (°C)","Pres (Pa)"])
# Graficar datos
x = np.linspace(0, 10, 100)
ax[0].plot(x, np.sin(x), label="Seno")
ax[1].plot(x, np.cos(x), label="Coseno")
# Configurar, mostrar y guardar
fig, ax = g.render(l_cols=1, ruta_guardado="Ejemplo/Figura1.png")
```