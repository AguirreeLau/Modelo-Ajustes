from typing import Callable, TypeAlias, Literal, Optional, Tuple, List
from scipy.odr import ODR, Model, RealData  ##
from odrpack import odr_fit
from ._decoradores import excepciones
from dataclasses import dataclass
from uncertainties import ufloat
import numpy as np

FuncionExplicita: TypeAlias = Callable[[np.ndarray, np.ndarray], np.ndarray]
FuncionImplicita: TypeAlias = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]

Funcion: TypeAlias = FuncionExplicita | FuncionImplicita

@dataclass
class Funciones:                                                    # Clase de las funciones a ajustar
    funcion: Funcion
    tipo: Literal["explicita", "implicita"] = "explicita"

    def __str__(self) -> str:
        func_name = getattr(self.funcion, "__name__", "<función anónima>")
        doc = self.funcion.__doc__
        if doc:
            return f"Función {func_name}:\n{doc.strip()}"
        else:
            return f"Función {func_name}: Sin descripción disponible."

    @excepciones(critico=True, imprimir=True)
    def fit_odr(self, xdata, ydata, beta0, *,
                errx=None, erry=None, errx_min=None, erry_min=None, estimadores = None,
                **kwargs) -> Tuple[List[ufloat], List[Optional[np.ndarray]]]:

        if errx is None and errx_min is not None:
            raise ValueError("No se puede definir errx_min sin errx.")
        if erry is None and erry_min is not None:
            raise ValueError("No se puede definir erry_min sin erry.")
        if "weight_x" in kwargs or "weight_y" in kwargs:
            raise ValueError("No se deben pasar pesos directamente. Use errx/erry.")
        if beta0 is None or len(beta0) == 0:
            raise ValueError("Insertar parámetros iniciales de busqueda")
        self._check_array(xdata, ydata)
        if errx is not None:
            self._check_array(xdata, errx)
        if erry is not None:
            self._check_array(ydata, erry)

        wx = self._peso(errx, errx_min) if errx is not None else None
        wy = self._peso(erry, erry_min) if erry is not None else None

        sol = odr_fit(self.funcion, xdata, ydata, beta0, weight_x=wx, weight_y=wy, **kwargs)
        beta_opt = [ufloat(val, err) for val, err in zip(sol.beta, sol.sd_beta)]

        estimadores_resultados = self._calcular_estimadores(estimadores, sol, xdata, ydata)

        from .fit_result import FitResult
        return sol, beta_opt, estimadores_resultados        # FALTA CAMBIAR EL RETURN PARA QUE DEVUELVA UN OBJETO DE LA CLASE FitResult CON TODOS LOS ESTIMADORES EN UN DICT COMÚN.
        # return FitResult(ODR_output=sol, parametros=beta_opt, estimadores=estimadores_resultados)


    def _check_array(self, a: np.ndarray, b: np.ndarray) -> None:
        """
        Verifica que dos arrays tengan la misma longitud.

        Args:
            -a (np.ndarray): Primer array a comparar.
            -b (np.ndarray): Segundo array a comparar.

        Retorna:
            -ValueError: Si los arrays no tienen la misma longitud.
        """
        if len(a) != len(b):
            raise ValueError(f"Los arrays deben tener la misma longitud. Se obtuvo {len(a)} y {len(b)} respectivamente.")
    
    @excepciones(critico=True, imprimir=True)
    def _peso(self, err: np.ndarray, err_min: Optional[float] = None) -> np.ndarray:
            if err_min is not None:
                err_eff = np.maximum(err, err_min)
            else:
                err_eff = err
            return  1.0 / err_eff**2

    def _calcular_estimadores(self, estimadores, sol, xdata, ydata):
        """
        Llama a las funciones individuales de cada estimador solicitado
        y devuelve un diccionario con los resultados.
        """
        if estimadores is None:
            return {}

        disponibles = {
            "R2": self._calc_r2,
            "R2 ajustado": self._calc_r2_ajustado,
            "Residuos": self._calc_residuos,
            "Chi2 reducido": self._calc_chi2_reducido,
            "Matriz de correlación": self._calc_matriz_correlacion,
        }
        if estimadores is True:
            seleccion = disponibles.keys()
        elif isinstance(estimadores, (list, tuple, set)):
            seleccion = estimadores
        else:
            raise TypeError("estimadores debe ser None, True o lista de nombres.")

        resultados = {}
        for nombre in seleccion:
            if nombre not in disponibles:
                raise ValueError(f"Estimador desconocido: {nombre}")
            resultados[nombre] = disponibles[nombre](sol, xdata, ydata)
        return resultados

    def _calc_residuos(self, sol, xdata, ydata):
        return ydata - self.funcion(sol.beta, xdata)

    def _calc_r2(self, sol, xdata, ydata):
        residuos = self._calc_residuos(sol, xdata, ydata)
        SCT = np.sum((ydata - np.mean(ydata))**2)
        SCR = np.sum(residuos**2)
        return 1 - (SCR / SCT)

    def _calc_r2_ajustado(self, sol, xdata, ydata):
        R2 = self._calc_r2(sol, xdata, ydata)
        n = len(ydata)
        p = len(sol.beta)
        return 1 - (1 - R2)*(n - 1)/(n - p - 1)

    def _calc_chi2_reducido(self, sol, xdata=None, ydata=None):
        chi2 = sol.sum_square
        dof = len(sol.eps) - len(sol.beta)
        return chi2 / dof if dof > 0 else np.inf

    def _calc_matriz_correlacion(self, sol, xdata=None, ydata=None):
        sd = sol.sd_beta
        cov = sol.cov_beta
        rv = sol.res_var
        return (cov * rv) / np.outer(sd, sd)

### Funciones comunes
    @staticmethod
    def polinomio(x, params):
        """
        Polinomio de grado n: y = a_n*x^n + ... + a_1*x + a_0

        Parámetros:
        params (array-like): Coeficientes del polinomio.
            a_0: Término independiente.
            a_1: Coeficiente del término lineal.
            ...
            a_n: Coeficiente del término de mayor grado.
        x (np.ndarray): Variable independiente.
        """
        return sum(b * np.power(x, i) for i, b in enumerate(params))

    @staticmethod
    def APV (x, params):
        """
        Módelo Pseudo-Voigt asimétrico para el ajuste picos. 
        Sean G(x) y L(x) una función Gaussiana y Lorentziana respectivamente, el modelo PV (Pseudo-Voigt) considera las contribuciones de ambas
        funciones bajo un parámetro eta que describe las constribuciones de cada una.
        A este modelo se le agrega la posibilidad de un comportamiento asimétrico respecto a su centro (x0).
        
        Parámetros:
            params (array-like): Array de los parámetros libres. 1 coresponde a x < x0 y 2 a x >= x0
                B[0] = A                --> Amplitud de la función.
                B[1] = x0               --> Posición de la media.
                B[2/4] = sigma_1/2      --> Desviación estandar de la función.
                B[3/5] = eta_1/2        --> Contribución en la suma de la parte Gaussiana, (eta-1) es la contribución de la Lorentziana.
                B[6] = y0               --> Corrimiento vertical de la función.
            x (np.ndarray): Variable independiente.
        """
        A, x0, sigma_1, eta_1, sigma_2, eta_2, y0 = params
        eta_1 = np.clip(eta_1, 0, 1)
        eta_2 = np.clip(eta_2, 0, 1)
        def G(sigma, x0, x):
            return np.exp(-((x-x0)/sigma)**2/2)
        def L(sigma, x0, x):
            return 1 /(1 + ((x-x0)/sigma)**2)
        return np.piecewise(x, 
            [x <= x0, x >= x0], 
            [
                lambda x: A*(eta_1*G(sigma_1, x0, x) + (1-eta_1)*L(sigma_1, x0, x)) + y0, 
                lambda x: A*(eta_2*G(sigma_2, x0, x) + (1-eta_2)*L(sigma_2, x0, x)) + y0
            ]
        )