import pytest
import numpy as np
from uncertainties.core import UFloat
from uncertainties import ufloat
from Base.FitResult import FitResult
from Base.Funciones import Funciones

# --- Fixtures de datos ---

@pytest.fixture
def datos_lineales():
    x = np.arange(0, 11, dtype=float)
    y = 2 * x + 1
    rng = np.random.default_rng(seed=1234)
    y = y + rng.normal(0, 0.05, size=len(x))  # ruido con sigma=0.05
    return x, y

@pytest.fixture
def func_lineal():
    def linea(params, x):
        a, b = params
        return a*x + b
    return Funciones(linea)

@pytest.fixture
def func_polinomial():
    return Funciones(Funciones.polinomio)

# --- Tests Funciones.fit_odr básico ---

def test_fit_odr_basico(func_lineal, datos_lineales):
    x, y = datos_lineales
    p0 = [0, 0]
    resultado = func_lineal.fit_odr(x, y, p0)

    # Validaciones generales
    assert isinstance(resultado, FitResult)
    assert len(resultado.parametros) == 2
    assert all(isinstance(p, UFloat) for p in resultado.parametros)
    assert 0.99 < resultado.R2 <= 1.0
    assert 0.99 < resultado.R2_aj <= 1.0

def test_fit_odr_con_errores(func_lineal, datos_lineales):
    x, y = datos_lineales
    p0 = [0, 0]
    err_x = np.full_like(x, 0.1)
    err_y = np.full_like(y, 0.2)
    resultado = func_lineal.fit_odr(x, y, p0, err_x=err_x, err_y=err_y)
    
    assert isinstance(resultado, FitResult)
    assert len(resultado.residuos) == len(x)

# --- Tests FitResult.__str__ y __iter__ ---

def test_str_iter(func_lineal, datos_lineales):
    x, y = datos_lineales
    p0 = [0, 0]
    resultado = func_lineal.fit_odr(x, y, p0)
    
    s = str(resultado)
    assert "p1" in s and "p2" in s
    params, R2, R2_aj, residuos, output = list(resultado)
    assert params == resultado.parametros
    assert R2 == resultado.R2
    assert R2_aj == resultado.R2_aj
    assert np.all(residuos == resultado.residuos)
    assert output == resultado.ODR_output

# --- Tests Jackknife ---

def test_jackknife_basico(func_lineal, datos_lineales):
    x, y = datos_lineales
    p0 = [0, 0]
    resultado = func_lineal.fit_odr(x, y, p0)
    jk_params, fits_jk = resultado.jackknife(func_lineal, x, y, p0=p0)
    
    assert len(jk_params) == len(p0)
    assert all(isinstance(p, UFloat) for p in jk_params)
    assert len(fits_jk) == len(x)

def test_jackknife_excluir_incluir_error(func_lineal, datos_lineales):
    x, y = datos_lineales
    p0 = [0, 0]
    resultado = func_lineal.fit_odr(x, y, p0)
    with pytest.raises(ValueError):
        resultado.jackknife(func_lineal, x, y, excluir=[0], incluir=[1])

def test_jackknife_objeto_invalido(datos_lineales):
    x, y = datos_lineales
    parametros = [ufloat(1.0, 0.1), ufloat(2.0, 0.2)]
    from types import SimpleNamespace
    ODR_output = SimpleNamespace(stopreason="Convergencia")
    resultado = FitResult(ODR_output=ODR_output, parametros=parametros)
    
    with pytest.raises(TypeError):
        resultado.jackknife(object(), x, y)

# --- Tests Funciones polinomio ---

def test_polinomio():
    x = np.array([0, 1, 2])
    params = [1, 2, 3]  # y = 1 + 2*x + 3*x^2
    y_calc = Funciones.polinomio(params, x)
    np.testing.assert_allclose(y_calc, [1, 6, 17])

# pytest Tests\\Test_Fit.py