# Modelo_Ajustes

Repositorio con funciones básicas para ajustes de datos con incertidumbres asociadas y visualización de resultados.

## Estructura

- `Base/` : Módulos principales
    - `Data.py` : Manejo de DataFrames y conversión a arrays con incertidumbres
    - `Funciones.py` : Funciones matemáticas y ajuste ODR
    - `FitResult.py` : Clase para almacenar resultados de ajuste
    - `Graficos.py` : Clase para generar figuras y subplots
    - `Decoradores.py` : Decoradores para manejo de excepciones

- `Datos/` : Datos de ejemplo (`demo_cubica.csv`)
- `Imagenes/` : Gráficos generados (`Ajuste cúbico.png`)
- `Tests/` : Tests unitarios para cada módulo
- `Notas/` : Documentación de cada módulo en markdown
- `main.py` : Demo completa del proyecto

## Instalación

```bash
git clone https://github.com/tu_usuario/Modelo_Ajustes.git
cd Modelo_Ajustes
pip install -r requirements.txt
