import pytest
import numpy as np
from fittools.fit_result import FitResult
from fittools.funciones import Funciones
from uncertainties.core import UFloat
from uncertainties import ufloat
from types import SimpleNamespace

# --- Fixtures ---

@pytest.fixture
def fit_result_mock():
    parametros = [ufloat(1.0, 0.1), ufloat(2.0, 0.2)]
    ODR_output = SimpleNamespace(stopreason="Convergencia")
    R2, R2_aj = 0.95, 0.93
    residuos = np.array([0.1, -0.1])
    return FitResult(ODR_output=ODR_output, parametros=parametros, R2=R2, R2_aj=R2_aj, residuos=residuos)

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

def test_str(fit_result_mock):
    s = str(fit_result_mock)
    assert "p1 = 1" in s
    assert "R² = 0.9500" in s
    assert "R² ajustado = 0.9300" in s
    assert "Convergencia" in s

def test_iter(fit_result_mock):
    params, R2, R2_aj, residuos, output = list(fit_result_mock)
    assert params == fit_result_mock.parametros
    assert R2 == fit_result_mock.R2
    assert R2_aj == fit_result_mock.R2_aj
    assert np.all(residuos == fit_result_mock.residuos)
    assert output == fit_result_mock.ODR_output

def test_jackknife_excluir_incluir_error(fit_result_mock):
    f_mock = FitResult
    x = np.array([1, 2, 3])
    y = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        fit_result_mock.jackknife(f_mock, x, y, excluir=[0], incluir=[1])

def test_jackknife_objeto_invalido(fit_result_mock):
    x = np.array([1, 2, 3])
    y = np.array([1, 2, 3])
    with pytest.raises(TypeError):
        fit_result_mock.jackknife(object(), x, y)

def test_jackknife_basico(func_lineal, datos_lineales):
    x, y = datos_lineales
    p0 = [0, 0]
    resultado = func_lineal.fit_odr(x, y, p0)
    jk_params, fits_jk = resultado.jackknife(func_lineal, x, y, p0=p0)
    assert len(jk_params) == len(p0)
    assert all(isinstance(p, UFloat) for p in jk_params)

# pytest Tests\\Test_FitResult.py