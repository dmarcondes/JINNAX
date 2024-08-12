from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'Interface to train Physics-informed Neural Networks in JAX'
LONG_DESCRIPTION = 'Interface to train Physics-informed Neural Networks in JAX'

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="jinnax",
        version=VERSION,
        author="Diego Marcondes",
        author_email="<dmarcondes@ime.usp.br>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=['pandas','openpyxl','jax','pillow','optax','alive_progress','IPython','numpy','matplotlib','dill'],

        keywords=['python', 'JAX', 'PINN'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Researchers",
            "Programming Language :: Python :: 3",
            "Operating System :: Linux :: Ubuntu",
        ]
)
