import pytest
import pandas as pd
import numpy as np
from Base import DF
from uncertainties import unumpy as unp
from pathlib import Path

# --- Fixtures ---

@pytest.fixture
def csv_path(tmp_path):
    """Crea un CSV de prueba temporal."""
    path = tmp_path / "datos.csv"
    df = pd.DataFrame({
        "x": [1, 2, 3],
        "y": [4, 5, 6],
        "z_err": [0.1, 0.2, 0.3]
    })
    df.to_csv(path, index=False)
    return path

@pytest.fixture
def df_instance(csv_path):
    """Instancia de DataFrame desde CSV de prueba."""
    return DF.desde_csv(csv_path, separacion=",", encabezados=True)

# --- Tests ---

def test_desde_csv(df_instance):
    """Verifica que la instancia cargue correctamente el DataFrame."""
    assert df_instance.df is not None
    assert list(df_instance.df.columns) == ["x", "y", "z_err"]

def test_filtrar_str(df_instance):
    """Filtrado usando string estilo query."""
    df_f = df_instance.filtrar("x > 1")
    assert all(df_f.df["x"] > 1)

def test_filtrar_series(df_instance):
    """Filtrado usando pd.Series booleana."""
    mask = pd.Series([True, False, True])
    df_f = df_instance.filtrar(mask)
    assert list(df_f.df.index) == [0, 2]

def test_filtrar_callable(df_instance):
    """Filtrado usando callable."""
    df_f = df_instance.filtrar(lambda df: df["y"] > 4)
    assert all(df_f.df["y"] > 4)

def test_separar_c_f(df_instance):
    """Separar por filas y columnas."""
    filas = df_instance.separar_c_f("f")
    columnas = df_instance.separar_c_f("c")
    assert len(filas) == len(df_instance.df)
    assert len(columnas) == len(df_instance.df.columns)
    assert all(isinstance(f, np.ndarray) for f in filas)
    assert all(isinstance(c, np.ndarray) for c in columnas)

def test_norm_str():
    """Prueba de normalización de strings."""
    s = "Temperatura (°C)"
    norm = DF._norm_str(s)
    assert norm == "temperatura_c"

def test_uarray_sufijo(df_instance):
    """Convierte columnas con sufijo de error a uarray."""
    df_instance.df["z"] = [10, 20, 30]  # columna nominal
    arrays = df_instance.uarray(str_="err", caso="sufijo")
    assert "z" in arrays
    assert unp.nominal_values(arrays["z"]).tolist() == [10, 20, 30]

def test_grilla_y_grad(df_instance):
    """Prueba de grilla y gradientes."""
    X, Y, dx, dy = df_instance.grilla_y_grad()
    assert X.shape == df_instance.df.shape
    assert Y.shape == df_instance.df.shape
    assert dx.shape == df_instance.df.shape
    assert dy.shape == df_instance.df.shape

# pytest Tests\\Test_Data.py