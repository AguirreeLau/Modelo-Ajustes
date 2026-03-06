from pathlib import Path
import sys


# Prioriza el paquete local del repositorio sobre cualquier instalación previa.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
