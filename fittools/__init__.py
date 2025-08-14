"""
fittools — Herramientas para ajuste de modelos y análisis de datos

Este módulo proporciona una interfaz unificada para acceder a los submódulos de `fittools`,
facilitando su importación y uso en proyectos de análisis y modelado.

---

Puedes importar los módulos de `fittools` de las siguientes maneras:

### 1. Importación completa del paquete
```python
import fittools
```

### 2. Importación de submódulos específicos
```python
import fittools.data_frames
import fittools.funciones
import fittools.graficos
import fittools.fit_result
```

### 3. Importación de submódulos específicos
```python
from fittools.data_frames import DataFrame
from fittools.funciones import Funciones
from fittools.graficos import Graficos
from fittools.fit_result import FitResult
```

### 4. Importación completa del paquete
```python
from fittools import *   # DataFrame as dframes, Funciones as funcs, FitResult as fres, Graficos as grafs
```
"""
from . import funciones, graficos, data_frames, fit_result
from .data_frames import DataFrame as dframes
from .funciones import Funciones as funcs
from .fit_result import FitResult as fres
from .graficos import Graficos as grafs

__all__ = ["dframes", "funcs", "fres", "grafs"]