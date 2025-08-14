import pytest
import numpy as np
from fittools.funciones import Funciones 
from uncertainties import ufloat

# --- Fixtures ---

@pytest.fixture
def datos_lineales():
    x = np.arange(0, 11, dtype=float)
    y = 2 * x + 1
    rng = np.random.default_rng(seed=1234)
    y = y + rng.normal(0, 0.05, size=len(x))  # ruido con sigma=0.05
    return x, y

@pytest.fixture
def func_lineal():
    return Funciones(lambda p, x: p[0]*x + p[1])

# --- Tests ---

def test_check_array_igual(func_lineal):
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    func_lineal._check_array(a, b)  # No debería lanzar excepción

def test_check_array_distinto(func_lineal):
    a = np.array([1, 2])
    b = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        func_lineal._check_array(a, b)

def test_coef_determinacion(func_lineal):
    y_obs = np.array([1, 2, 3])
    residuos = np.array([0, 0, 0])
    R2, R2_aj = func_lineal._coef_determinacion(y_obs, residuos, cant_params=1)
    assert R2 == 1.0
    assert R2_aj == 1.0

def test_fit_odr_lineal(func_lineal, datos_lineales):
    x, y = datos_lineales
    p0 = [1, 0]  # Parámetros iniciales
    resultado = func_lineal.fit_odr(x, y, p0)
    # Verifica que los parámetros sean aproximados a [2, 1]
    p_opt = [p.n for p in resultado.parametros]
    assert pytest.approx(p_opt[0], rel=1e-2) == 2
    assert pytest.approx(p_opt[1], rel=1e-2) == 1

def test_polinomio():
    x = np.array([0, 1, 2])
    y = Funciones.polinomio([1, 2, 3], x)  # 3*x^2 + 2*x + 1
    assert np.allclose(y, [1, 6, 17])

def test_APV_salida():
    x = np.linspace(-1, 1, 5)
    params = [1, 0, 1, 0.5, 1, 0.5, 0]  # valores arbitrarios
    y = Funciones.APV(params, x)
    assert y.shape == x.shape
    assert np.all(np.isfinite(y))

# pytest Tests\\Test_Funciones.py