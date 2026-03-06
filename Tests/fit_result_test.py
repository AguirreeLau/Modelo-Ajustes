import numpy as np
import pytest

from fittools.fit_result import FitResult
from fittools.funciones import Funciones


@pytest.fixture
def resultado_explicito():
    def linea(x, params):
        return params[0] * x + params[1]

    modelo = Funciones(linea)
    x = np.linspace(0.0, 10.0, 25)
    y = 3.0 * x - 1.0
    return modelo.fit_odr(x, y, beta0=[0.0, 0.0], estimadores=True)


@pytest.fixture
def resultado_implicito():
    modelo = Funciones(Funciones.circunferencia)
    t = np.linspace(0.0, 2.0 * np.pi, 60, endpoint=False)
    x = 2.0 + 4.0 * np.cos(t)
    y = -1.0 + 4.0 * np.sin(t)
    return modelo.fit_odr(x, y, beta0=[1.0, 0.0, 3.5], estimadores=True)


def test_fitresult_tipo(resultado_explicito):
    assert isinstance(resultado_explicito, FitResult)
    assert resultado_explicito.odrresult is not None
    assert len(resultado_explicito.parametros) == 2


def test_str_contiene_campos_clave(resultado_explicito):
    s = str(resultado_explicito)
    assert "Parámetros" in s
    assert "p1" in s
    assert "Motivo(s) de finalización" in s


def test_iter_explicito_orden(resultado_explicito):
    values = list(resultado_explicito)
    assert len(values) == 7

    params, r2, r2_aj, residuos, chi2, mcor, odr = values
    assert params == resultado_explicito.parametros
    assert r2 == resultado_explicito.estimadores.get("R2")
    assert r2_aj == resultado_explicito.estimadores.get("R2 ajustado")
    assert np.array_equal(residuos, resultado_explicito.estimadores.get("Residuos"))
    assert chi2 == resultado_explicito.estimadores.get("Chi2 reducido")
    assert np.array_equal(mcor, resultado_explicito.estimadores.get("Matriz de correlación"))
    assert odr == resultado_explicito.odrresult


def test_iter_implicito_r2_y_residuos_son_none(resultado_implicito):
    _, r2, r2_aj, residuos, chi2, mcor, _ = list(resultado_implicito)
    assert r2 is None
    assert r2_aj is None
    assert residuos is None
    assert chi2 is not None
    assert mcor is not None


def test_jackknife_error_excluir_e_incluir(resultado_explicito):
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([2.0, 4.0, 6.0, 8.0])
    with pytest.raises(ValueError):
        resultado_explicito.jackknife(
            object(),
            x,
            y,
            excluir=[0],
            incluir=[1],
        )


def test_jackknife_error_objeto_invalido(resultado_explicito):
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([2.0, 4.0, 6.0])
    with pytest.raises(TypeError):
        resultado_explicito.jackknife(object(), x, y)


def test_jackknife_basico_funciona(resultado_explicito):
    def linea(x, params):
        return params[0] * x + params[1]

    modelo = Funciones(linea)
    x = np.linspace(0.0, 5.0, 10)
    y = 2.0 * x + 1.0 + np.linspace(-0.01, 0.01, x.size)

    params_jk, fits_jk = resultado_explicito.jackknife(modelo, x, y, p0=[0.0, 0.0])
    assert len(params_jk) == 2
    assert len(fits_jk) == len(x)


def test_jackknife_con_errores_funciona(resultado_explicito):
    def linea(x, params):
        return params[0] * x + params[1]

    modelo = Funciones(linea)
    x = np.linspace(0.0, 5.0, 12)
    y = 1.5 * x + 0.2 + np.linspace(-0.02, 0.02, x.size)
    err_x = np.full_like(x, 0.05)
    err_y = np.full_like(x, 0.08)

    params_jk, fits_jk = resultado_explicito.jackknife(
        modelo,
        x,
        y,
        p0=[0.0, 0.0],
        err_x=err_x,
        err_y=err_y,
    )
    assert len(params_jk) == 2
    assert len(fits_jk) == len(x)
