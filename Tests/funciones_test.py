import numpy as np
import pytest

from fittools.fit_result import FitResult
from fittools.funciones import Funciones


@pytest.fixture
def datos_lineales():
    x = np.linspace(0.0, 10.0, 30)
    rng = np.random.default_rng(seed=1234)
    y = 2.0 * x + 1.0 + rng.normal(0.0, 0.05, size=x.size)
    return x, y


@pytest.fixture
def modelo_lineal():
    def linea(x, params):
        return params[0] * x + params[1]

    return Funciones(linea)


@pytest.fixture
def datos_circunferencia():
    t = np.linspace(0.0, 2.0 * np.pi, 80, endpoint=False)
    x = 1.5 + 3.0 * np.cos(t)
    y = -2.0 + 3.0 * np.sin(t)
    return x, y


def test_modelo_implicito_infiere_tipo():
    modelo = Funciones(Funciones.circunferencia)
    assert modelo.tipo == "implicita"


def test_check_array_igual(modelo_lineal):
    modelo_lineal._check_array(np.array([1, 2, 3]), np.array([4, 5, 6]))


def test_check_array_distinto(modelo_lineal):
    with pytest.raises(ValueError):
        modelo_lineal._check_array(np.array([1, 2]), np.array([1, 2, 3]))


def test_peso_con_minimo(modelo_lineal):
    err = np.array([0.01, 0.20, 0.05])
    w = modelo_lineal._peso(err, err_min=0.10)
    np.testing.assert_allclose(w, 1.0 / np.array([0.10, 0.20, 0.10]) ** 2)


def test_fit_odr_explicito_devuelve_fitresult(modelo_lineal, datos_lineales):
    x, y = datos_lineales
    resultado = modelo_lineal.fit_odr(x, y, beta0=[0.0, 0.0], estimadores=True)

    assert isinstance(resultado, FitResult)
    assert len(resultado.parametros) == 2
    assert resultado.estimadores["R2"] > 0.99
    assert "Residuos" in resultado.estimadores


def test_fit_odr_implicito_estimadores(datos_circunferencia):
    x, y = datos_circunferencia
    modelo = Funciones(Funciones.circunferencia)

    resultado = modelo.fit_odr(x, y, beta0=[1.0, -1.0, 2.0], estimadores=True)
    delta = np.asarray(resultado.estimadores["delta"])
    modulo = np.asarray(resultado.estimadores["modulo_delta"])

    assert isinstance(resultado, FitResult)
    assert sorted(resultado.estimadores.keys()) == [
        "Chi2 reducido",
        "Matriz de correlación",
        "delta",
        "modulo_delta",
    ]
    assert delta.shape == (2, x.size)
    np.testing.assert_allclose(modulo, np.sqrt(delta[0] ** 2 + delta[1] ** 2))


def test_fit_odr_implicito_no_acepta_r2(datos_circunferencia):
    x, y = datos_circunferencia
    modelo = Funciones(Funciones.circunferencia)

    with pytest.raises(ValueError):
        modelo.fit_odr(x, y, beta0=[1.0, -1.0, 2.0], estimadores=["R2"])


def test_validaciones_entrada(modelo_lineal, datos_lineales):
    x, y = datos_lineales

    with pytest.raises(ValueError):
        modelo_lineal.fit_odr(x, y, beta0=[0.0, 0.0], errx_min=0.1)

    with pytest.raises(ValueError):
        modelo_lineal.fit_odr(x, y, beta0=[0.0, 0.0], erry_min=0.1)

    with pytest.raises(ValueError):
        modelo_lineal.fit_odr(x, y, beta0=None)

    with pytest.raises(ValueError):
        modelo_lineal.fit_odr(x, y, beta0=[0.0, 0.0], weight_x=np.ones_like(x))


def test_warn_si_solo_hay_error_en_un_eje(modelo_lineal, datos_lineales):
    x, y = datos_lineales
    err = np.full_like(x, 0.2)

    with pytest.warns(UserWarning):
        modelo_lineal.fit_odr(x, y, beta0=[0.0, 0.0], errx=err)

    with pytest.warns(UserWarning):
        modelo_lineal.fit_odr(x, y, beta0=[0.0, 0.0], erry=err)


def test_polinomio():
    x = np.array([0.0, 1.0, 2.0])
    params = [1.0, 2.0, 3.0]  # 1 + 2x + 3x^2
    y = Funciones.polinomio(x, params)
    np.testing.assert_allclose(y, [1.0, 6.0, 17.0])


def test_apv_salida():
    x = np.linspace(-1.0, 1.0, 11)
    params = [1.0, 0.0, 1.0, 0.5, 1.0, 0.5, 0.0]
    y = Funciones.APV(x, params)
    assert y.shape == x.shape
    assert np.all(np.isfinite(y))
