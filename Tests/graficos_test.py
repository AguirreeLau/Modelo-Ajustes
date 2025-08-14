import pytest
from pathlib import Path
from fittools import grafs

def test_instancia_graficos():
    """Verifica que se pueda crear una instancia y combinar colores/fontsizes."""
    g = grafs(columnas=2, colores={"titulo": "#FF0000"}, fontsizes={"titulo": 20})

    # Comprobar que los valores por defecto se mantienen para keys no pasadas
    assert g.colores["titulo"] == "#FF0000"
    assert g.colores["eje_x"] == "#1F2020"
    assert g.fontsizes["titulo"] == 20
    assert g.fontsizes["eje_x"] == 12

def test_crear_figura_y_axes():
    """Verifica que crear() devuelva figuras y axes correctamente."""
    titulos = ["T1", "T2"]
    g = grafs(columnas=2)
    fig, axes = g.crear(titulo=titulos, eje_x="x", eje_y="y")

    # Debe devolver objeto Figure
    assert fig is not None
    # Debe devolver lista de Axes del tamaño correcto
    assert len(axes) == 2

    # Verificar títulos usando el atributo de la instancia
    for i, titulo in enumerate(titulos):
        assert g._titulos_asignados[i] == titulo, f"El título {titulo} no se encuentra en {g._titulos_asignados[i]}"

def test_render_limites_y_guardado(tmp_path):
    """Verifica que render() aplique límites y guarde la figura correctamente."""
    import numpy as np

    # Crear instancia del gráfico
    g = grafs(columnas=1)
    fig, ax = g.crear(titulo="Prueba", eje_x="X", eje_y="Y")
    
    # Datos de prueba
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    ax.plot(x, y, label="sin(x)")

    # Limites de prueba
    limites_x = (0, 10)
    limites_y = (0, 100)

    # Ruta de guardado temporal
    ruta_guardado = tmp_path / "grafico_prueba.png"
    fig, ax = g.render(l_cols=1, limite_x=limites_x, limite_y=limites_y, mostrar=False, ruta_guardado=ruta_guardado)
    
    # Revisar límites
    assert ax.get_xlim() == limites_x
    assert ax.get_ylim() == limites_y
    # Revisar que se haya guardado el archivo
    assert ruta_guardado.exists()

# pytest Tests\\Test_Graficos.py