import numpy as np
import pytest

from fittools.fit_result import FitResult
from fittools.funciones import Funciones


@pytest.fixture
def datos_lineales():
    x = np.linspace(0.0, 8.0, 40)
    rng = np.random.default_rng(seed=2026)
    y = 1.75 * x - 0.3 + rng.normal(0.0, 0.03, size=x.size)
    return x, y


@pytest.fixture
def modelo_lineal():
    def linea(x, params):
        m, b = params
        return m * x + b

    return Funciones(linea, tipo="explicita")


@pytest.fixture
def datos_circunferencia():
    t = np.linspace(0.0, 2.0 * np.pi, 120, endpoint=False)
    x = -1.0 + 2.5 * np.cos(t)
    y = 0.5 + 2.5 * np.sin(t)
    return x, y


def test_fit_odr_explicito_parametros_cercanos(modelo_lineal, datos_lineales):
    x, y = datos_lineales
    resultado = modelo_lineal.fit_odr(x, y, beta0=[0.0, 0.0], estimadores=True)

    assert isinstance(resultado, FitResult)
    m_fit, b_fit = [p.nominal_value for p in resultado.parametros]
    assert m_fit == pytest.approx(1.75, rel=1e-2)
    assert b_fit == pytest.approx(-0.3, abs=0.07)


def test_fit_odr_implicito_parametros_cercanos(datos_circunferencia):
    x, y = datos_circunferencia
    modelo = Funciones(Funciones.circunferencia)  # tipo inferido a implicita
    resultado = modelo.fit_odr(x, y, beta0=[-0.5, 0.0, 2.0], estimadores=True)

    h_fit, k_fit, r_fit = [p.nominal_value for p in resultado.parametros]
    assert h_fit == pytest.approx(-1.0, abs=0.02)
    assert k_fit == pytest.approx(0.5, abs=0.02)
    assert abs(r_fit) == pytest.approx(2.5, abs=0.02)


def test_fit_odr_seleccion_estimadores_explicito(modelo_lineal, datos_lineales):
    x, y = datos_lineales
    resultado = modelo_lineal.fit_odr(
        x,
        y,
        beta0=[0.0, 0.0],
        estimadores=["R2", "Chi2 reducido"],
    )

    assert sorted(resultado.estimadores.keys()) == ["Chi2 reducido", "R2"]


def test_fit_odr_seleccion_estimadores_implicito(datos_circunferencia):
    x, y = datos_circunferencia
    modelo = Funciones(Funciones.circunferencia)
    resultado = modelo.fit_odr(
        x,
        y,
        beta0=[-0.5, 0.0, 2.0],
        estimadores=["delta", "modulo_delta"],
    )

    assert sorted(resultado.estimadores.keys()) == ["delta", "modulo_delta"]
    assert np.asarray(resultado.estimadores["delta"]).shape == (2, x.size)


def test_fit_odr_error_shape_en_incertidumbre(modelo_lineal, datos_lineales):
    x, y = datos_lineales
    with pytest.warns(UserWarning):
        with pytest.raises(ValueError):
            modelo_lineal.fit_odr(x, y, beta0=[0.0, 0.0], errx=np.ones(3))


def test_tipo_invalido_en_constructor():
    with pytest.raises(ValueError):
        Funciones(Funciones.polinomio, tipo="otro")
