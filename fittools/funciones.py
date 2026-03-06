from typing import Callable, TypeAlias, Literal, Optional, Tuple, List
import warnings
from odrpack import odr_fit
from ._decoradores import excepciones
from .fit_result import FitResult
from dataclasses import dataclass
from uncertainties import ufloat
import numpy as np

def modelo_implicito(func):
    func.tipo = "implicita"
    return func

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
            **kwargs) -> FitResult:
        """
        Ejecuta un ajuste ODR para la funcion configurada en la instancia.

        El metodo valida la consistencia de los datos de entrada, construye los
        pesos a partir de las incertidumbres experimentales y ejecuta `odr_fit`.
        Luego empaqueta los parametros ajustados como `ufloat` y calcula
        estimadores adicionales opcionales.

        Args:
            xdata: Valores de la variable independiente.
            ydata: Valores de la variable dependiente.
            beta0: Estimacion inicial de los parametros libres.
            errx: Incertidumbre asociada a `xdata`.
            erry: Incertidumbre asociada a `ydata`.
            errx_min: Cota inferior para `errx` al construir los pesos.
            erry_min: Cota inferior para `erry` al construir los pesos.
            estimadores: None, True o coleccion con nombres de estimadores.
            **kwargs: Argumentos extra compatibles con `odr_fit`.

        Returns:
            FitResult con:
            - objeto OdrResult de odrpack (`odrresult`)
            - parametros ajustados con incertidumbre (`parametros`)
            - estimadores solicitados (`estimadores`)

        Raises:
            ValueError: Si faltan parametros iniciales, longitudes no coinciden o
                se pasan pesos directos por `weight_x`/`weight_y`.
        """    
        if errx is None and errx_min is not None:
            raise ValueError("No se puede definir errx_min sin errx.")
        if erry is None and erry_min is not None:
            raise ValueError("No se puede definir erry_min sin erry.")
        if errx is None and erry is not None:
            warnings.warn(
                "Solo se definieron incertidumbres en y. "
                "Se asumirá peso unitario para x."
            )
        if erry is None and errx is not None:
            warnings.warn(
                "Solo se definieron incertidumbres en x. "
                "Se asumirá peso unitario para y."
            )
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

        if self.tipo == "explicita":
            return self._ODR_explicito(xdata, ydata, beta0, wx=wx, wy=wy, estimadores=estimadores, **kwargs)
        else:
            return self._ODR_implicito(xdata, ydata, beta0, wx=wx, wy=wy, estimadores=estimadores, **kwargs)

# Diferenciación de modelos explícitos e implícitos
    def _ODR_explicito(self, xdata, ydata, beta0, *,
                wx=None, wy=None, estimadores = None,
                **kwargs) -> FitResult:
        """
        Ejecuta un ajuste ODR para la funcion configurada en la instancia.

        El metodo valida la consistencia de los datos de entrada, construye los
        pesos a partir de las incertidumbres experimentales y ejecuta `odr_fit`.
        Luego empaqueta los parametros ajustados como `ufloat` y calcula
        estimadores adicionales opcionales.

        Args:
            xdata: Valores de la variable independiente.
            ydata: Valores de la variable dependiente.
            beta0: Estimacion inicial de los parametros libres.
            errx: Incertidumbre asociada a `xdata`.
            erry: Incertidumbre asociada a `ydata`.
            errx_min: Cota inferior para `errx` al construir los pesos.
            erry_min: Cota inferior para `erry` al construir los pesos.
            estimadores: None, True o coleccion con nombres de estimadores.
            **kwargs: Argumentos extra compatibles con `odr_fit`.

        Returns:
            FitResult con:
            - objeto OdrResult de odrpack (`odrresult`)
            - parametros ajustados con incertidumbre (`parametros`)
            - estimadores solicitados (`estimadores`)

        Raises:
            ValueError: Si faltan parametros iniciales, longitudes no coinciden o
                se pasan pesos directos por `weight_x`/`weight_y`.
        """

        sol = odr_fit(self.funcion, xdata, ydata, beta0, weight_x=wx, weight_y=wy, task='explicit-ODR', **kwargs)
        beta_opt = [ufloat(val, err) for val, err in zip(sol.beta, sol.sd_beta)]

        estimadores_res = self._calcular_estimadores(estimadores, sol, xdata, ydata)

        return FitResult(odrresult=sol, parametros=beta_opt, estimadores=estimadores_res)

    def _ODR_implicito(self, xdata, ydata, beta0, *,
                wx=None, wy=None, estimadores = None,
                **kwargs) -> FitResult:
        """
        Ejecuta un ajuste ODR para la funcion configurada en la instancia.

        El metodo valida la consistencia de los datos de entrada, construye los
        pesos a partir de las incertidumbres experimentales y ejecuta `odr_fit`.
        Luego empaqueta los parametros ajustados como `ufloat` y calcula
        estimadores adicionales opcionales.

        Args:
            xdata: Valores de la variable independiente.
            ydata: Valores de la variable dependiente.
            beta0: Estimacion inicial de los parametros libres.
            errx: Incertidumbre asociada a `xdata`.
            erry: Incertidumbre asociada a `ydata`.
            errx_min: Cota inferior para `errx` al construir los pesos.
            erry_min: Cota inferior para `erry` al construir los pesos.
            estimadores: None, True o coleccion con nombres de estimadores.
            **kwargs: Argumentos extra compatibles con `odr_fit`.

        Returns:
            FitResult con:
            - objeto OdrResult de odrpack (`odrresult`)
            - parametros ajustados con incertidumbre (`parametros`)
            - estimadores solicitados (`estimadores`)

        Raises:
            ValueError: Si faltan parametros iniciales, longitudes no coinciden o
                se pasan pesos directos por `weight_x`/`weight_y`.
        """
        X = np.vstack([xdata, ydata])
        N = len(xdata)
        if wx is None and wy is None:
            wX = None
        elif wx is None:
            wX = np.vstack([np.ones(N), wy])
        elif wy is None:
            wX = np.vstack([wx, np.ones(N)])
        else:
            wX = np.vstack([wx, wy])

        array_nulo = np.zeros_like(xdata)

        sol = odr_fit(self.funcion, X, array_nulo, beta0, weight_x=wX, weight_y=None, task='implicit-ODR', **kwargs)
        beta_opt = [ufloat(val, err) for val, err in zip(sol.beta, sol.sd_beta)]

        estimadores_res = self._calcular_estimadores(estimadores, sol, xdata, ydata)

        return FitResult(odrresult=sol, parametros=beta_opt, estimadores=estimadores_res)

# Métodos internos para fit_odr
    @excepciones(critico=True, imprimir=True)
    def _check_array(self, a: np.ndarray, b: np.ndarray) -> None:
        """
        Verifica que dos arreglos tengan la misma longitud.

        Args:
            a (np.ndarray): Primer arreglo a comparar.
            b (np.ndarray): Segundo arreglo a comparar.

        Raises:
            ValueError: Si los arreglos no tienen la misma longitud.
        """
        if len(a) != len(b):
            raise ValueError(f"Los arrays deben tener la misma longitud. Se obtuvo {len(a)} y {len(b)} respectivamente.")

    @excepciones(critico=True, imprimir=True)
    def _peso(self, err: np.ndarray, err_min: Optional[float] = None) -> np.ndarray:
        """
        Calcula pesos estadisticos como inversa del error al cuadrado.

        Si `err_min` se define, cada componente de `err` se acota por debajo
        con ese valor para evitar pesos excesivos por errores muy pequenos.

        Args:
            err (np.ndarray): Incertidumbres experimentales.
            err_min (Optional[float]): Cota inferior para `err`.

        Returns:
            np.ndarray: Vector de pesos `1 / err_eff**2`.
        """
        if err_min is not None:
            err_eff = np.maximum(err, err_min)
        else:
            err_eff = err
        return 1.0 / err_eff**2

    @excepciones(critico=True, imprimir=True)
    def _calcular_estimadores(self, estimadores, sol, xdata, ydata):
        """
        Calcula un conjunto configurable de metricas del ajuste.

        Args:
            estimadores: None, True o coleccion de nombres de estimadores.
            sol: Resultado devuelto por ODR.
            xdata: Valores de la variable independiente.
            ydata: Valores observados de la variable dependiente.

        Returns:
            dict: Diccionario `nombre -> valor` con las metricas solicitadas.

        Raises:
            TypeError: Si `estimadores` no es None, True o coleccion valida.
            ValueError: Si se solicita un estimador no soportado.
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

    @excepciones(critico=True, imprimir=True)
    def _calc_residuos(self, sol, xdata, ydata):
        """
        Calcula los residuos del ajuste.

        Args:
            sol: Resultado devuelto por ODR.
            xdata: Variable independiente usada en el ajuste.
            ydata: Datos observados.

        Returns:
            np.ndarray: Diferencia entre datos observados y modelo evaluado.
        """
        return ydata - self.funcion(xdata, sol.beta)

    @excepciones(critico=True, imprimir=True)
    def _calc_r2(self, sol, xdata, ydata):
        """
        Calcula el coeficiente de determinacion R2.

        Args:
            sol: Resultado devuelto por ODR.
            xdata: Variable independiente usada en el ajuste.
            ydata: Datos observados.

        Returns:
            float: Valor de R2.
        """
        residuos = self._calc_residuos(sol, xdata, ydata)
        SCT = np.sum((ydata - np.mean(ydata))**2)
        SCR = np.sum(residuos**2)
        return 1 - (SCR / SCT)

    @excepciones(critico=True, imprimir=True)
    def _calc_r2_ajustado(self, sol, xdata, ydata):
        """
        Calcula el coeficiente de determinacion ajustado.

        Args:
            sol: Resultado devuelto por ODR.
            xdata: Variable independiente usada en el ajuste.
            ydata: Datos observados.

        Returns:
            float: Valor de R2 ajustado por cantidad de parametros.
        """
        R2 = self._calc_r2(sol, xdata, ydata)
        n = len(ydata)
        p = len(sol.beta)
        return 1 - (1 - R2)*(n - 1)/(n - p - 1)

    @excepciones(critico=True, imprimir=True)
    def _calc_chi2_reducido(self, sol, xdata=None, ydata=None):
        """
        Calcula el chi cuadrado reducido del ajuste.

        Args:
            sol: Resultado devuelto por ODR.
            xdata: No utilizado. Se conserva por compatibilidad de interfaz.
            ydata: No utilizado. Se conserva por compatibilidad de interfaz.

        Returns:
            float: `chi2 / grados_de_libertad` o `np.inf` si dof <= 0.
        """
        chi2 = sol.sum_square
        dof = len(sol.eps) - len(sol.beta)
        return chi2 / dof if dof > 0 else np.inf

    @excepciones(critico=True, imprimir=True)
    def _calc_matriz_correlacion(self, sol, xdata=None, ydata=None):
        """
        Calcula la matriz de correlacion entre parametros ajustados.

        Args:
            sol: Resultado devuelto por ODR.
            xdata: No utilizado. Se conserva por compatibilidad de interfaz.
            ydata: No utilizado. Se conserva por compatibilidad de interfaz.

        Returns:
            np.ndarray: Matriz de correlacion derivada de la covarianza.
        """
        sd = sol.sd_beta
        cov = sol.cov_beta
        rv = sol.res_var
        return (cov * rv) / np.outer(sd, sd)

### Funciones comunes
## Explicitas
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

## Implicitas
    @staticmethod
    @modelo_implicito
    def elipse(X, params):
        x, y = X[0], X[1]
        h, k, a, b = params
        return ((x - h)/a)**2 + ((y - k)/b)**2 - 1

    @staticmethod
    @modelo_implicito
    def circunferencia(X, params):
        x, y = X[0], X[1]
        h, k, r = params
        return (x - h)**2 + (y - k)**2 - r**2

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