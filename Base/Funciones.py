from typing import Callable, Optional, Tuple, List
from scipy.odr import ODR, Model, RealData
from .Decoradores import excepciones
from dataclasses import dataclass
from uncertainties import ufloat
import numpy as np

@dataclass
class Funciones:                                                    # Clase de las funciones a ajustar
    funcion: Callable[[List[float], np.ndarray], np.ndarray]

    def __str__(self) -> str:
        func_name = getattr(self.funcion, "__name__", "<función anónima>")
        doc = self.funcion.__doc__
        if doc:
            return f"Función {func_name}:\n{doc.strip()}"
        else:
            return f"Función {func_name}: Sin descripción disponible."

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
    def _coef_determinacion(self, data_y: np.ndarray, residuos: np.ndarray, cant_params: int) -> float:
        """
        Calcula el coeficiente de determinación R² y su versión ajustada del modelo.

        Args:
            -data_y (np.ndarray): Valores observados de la variable dependiente.
            -residuos (np.ndarray): Diferencias entre valores observados y valores predichos por el modelo.
            -cant_params (int): Número de parámetros libres usados en el modelo (para ajuste de R²).

        Retorna:
            Tuple[float, float]
                - R² (coeficiente de determinación).
                - R² ajustado, penaliza la complejidad del modelo considerando el número de parámetros.
        
        Notas:
            - R² mide la proporción de la varianza explicada por el modelo.
            - R² ajustado es más conservador y evita la sobreestimación al incluir más parámetros.
        """
        SCR = np.sum(residuos**2)                                   # Suma de cuadrados de los residuos   
        SCT = np.sum((data_y - np.mean(data_y))**2)                 # Suma de cuadrados totales
        R2 = 1 - (SCR / SCT)
        n = len(data_y)
        R2_aj = 1 - (1 - R2)*(n - 1)/(n - cant_params - 1)

        return R2, R2_aj

    @excepciones(critico=True, imprimir=True)
    def fit_odr(self, data_x: np.ndarray, data_y: np.ndarray, p0: list[float], err_x: Optional[np.ndarray]=None, err_y: Optional[np.ndarray]=None, estimadores: Optional[bool]=True, sstol: Optional[float]=None, partol: Optional[float]=None, maxit: Optional[int]=None, iprint: Optional[int]=None) -> Tuple[List[ufloat], List[Optional[np.ndarray]]]:
        """
        Ajusta un modelo a los datos utilizando la técnica de Orthogonal Distance Regression (ODR).

        Args:
            - data_x (np.ndarray): Datos de la variable independiente.
            - data_y (np.ndarray): Datos de la variable dependiente.
            - err_x (np.ndarray): Errores en la variable independiente.
            - err_y (np.ndarray): Errores en la variable dependiente.
            - p0 (list): Parámetros iniciales para el ajuste.
            - estimadores (bool): Si True, calcula los estimadores de ajuste (R², R² ajustado y residuos).
            - sstol (float): Tolerancia para convergencia en suma de cuadrados (default None, default de Scipy).
            - partol (float): Tolerancia para convergencia en parámetros (default None, default de Scipy).
            - maxit (int): Máximo número de iteraciones permitidas (default None, default de Scipy).
            - iprint (int): Control de impresión de ODR (default None, default de Scipy).
        Retorna:
            -Tuple:
                p_opt (list): Parámetros ajustados como "ufloat" (valor ± error).
                estimadores (list or None): Lista [R2, R2_aj, residuos] si estimadores es True, o [None, None, None].
        Notas:
            - Se utiliza ODR para ajustar el modelo a los datos, considerando errores en ambas variables.
            - Se toma sd_beta como representación del error en los prámetros. De necesitarse covarianzas, acceder a resultado.cov_beta.
            - sd_beta se calcula como la raíz cuadrada de la diagonal de la matriz de covarianza escalada por la varianza residual del ajuste.
            - Los parámetros ajustados se devuelven como "ufloat" para incluir incertidumbres.
        """
        if not p0 :
            raise ValueError("Insertar parámetros iniciales de busqueda")
        self._check_array(data_x, data_y)
        if err_x is not None:
            self._check_array(data_x, err_x)
        if err_y is not None:
            self._check_array(data_y, err_y)

        cant_params = len(p0)

        modelo = Model(self.funcion)
        datos = RealData(data_x, data_y, sx = err_x, sy = err_y)
        ajuste  = ODR(datos, modelo, beta0 = p0, sstol = sstol, partol = partol, maxit = maxit, iprint = iprint)

        resultado = ajuste.run()
        p_opt = [ufloat(b, err) for b, err in zip(resultado.beta, resultado.sd_beta)]

        R2, R2_aj, residuos = None, None, None
        if estimadores:
            residuos = data_y - self.funcion(resultado.beta, data_x)
            R2, R2_aj = self._coef_determinacion(data_y, residuos, cant_params)

        from .FitResult import FitResult
        return FitResult(ODR_output=resultado, parametros=p_opt, R2=R2, R2_aj=R2_aj, residuos=residuos)

### Funciones comunes
    @staticmethod
    def polinomio(params, x):
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
    def APV (params, x):
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