from setuptools import setup, find_packages

setup(
    name="Modelo_Ajustes",
    version="0.1.0",
    author="Lautaro Aguirre",
    author_email="aguirreelautaro@gmail.com",
    description="Funciones para ajustes de datos con incertidumbres y visualización de resultados.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/AguirreeLau/Modelo_Ajustes",
    packages=find_packages(),  # detecta automáticamente Base y otros paquetes con __init__.py
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "scipy",
        "uncertainties"
    ],
    python_requires=">=3.8",
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)