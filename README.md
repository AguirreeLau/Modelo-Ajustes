# Modelo_Ajustes

Repositorio con funciones básicas para ajustes de datos con incertidumbres asociadas y visualización de resultados.

## Estructura

- `Base/` : Módulos principales
    - `Data.py` : Manejo de DataFrames y conversión a arrays con incertidumbres.
    - `Funciones.py` : Funciones matemáticas y ajuste ODR.
    - `FitResult.py` : Clase para almacenar resultados de ajuste.
    - `Graficos.py` : Clase para generar figuras y subplots.
    - `Decoradores.py` : Decoradores para manejo de excepciones.

- `Datos/` : Datos de ejemplo (`demo_cubica.csv`).
- `Imagenes/` : Gráficos generados (`Ajuste cúbico.png`).
- `Tests/` : Tests unitarios para cada módulo.
- `Notas/` : Documentación de cada módulo en markdown.
- `main.py` : Demo completa del proyecto.

## Ejemplo de salida

```
################################
####  Resultado del ajuste  ####
################################
* Parámetros:
      - p1 = 0.4323 ± 0.96
      - p2 = 3.412 ± 0.55
      - p3 = -1.092 ± 0.085
      - p4 = 0.4551 ± 0.033
* R² = 0.9831
* R² ajustado = 0.9816
* Motivo(s) de finalización:
      - Sum of squares convergence
################################
```

![Ajuste cúbico](Imagenes/Ajuste_cubico.png)

## Instalación

```bash
git clone https://github.com/tu_usuario/Modelo_Ajustes.git
cd Modelo_Ajustes
pip install -r requirements.txt
```

## Estado actual

Actualmente el proyecto está en estado preliminar, pueden implementarse muchas mejoras en las herramientas de la carpeta Base, sin embargo, todas pueden utilizarse y aligeran bastantante el manejo de datos.
