from setuptools import setup, find_packages

setup(
    name="eoh",
    version="0.1",
    author="MetaAI Group, CityU",
    description="Evolutionary Computation + Large Language Model for automatic algorithm design",
    packages=find_packages(where="eoh/src"),
    package_dir={"": "eoh/src"},
    python_requires=">=3.10",
    install_requires=[
        "astunparse>=1.6",
        "numpy",
        "numba",
        "joblib",
        "networkx>=3.0",
        "requests>=2.31",
    ],
    test_suite="tests"
)
