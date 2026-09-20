from setuptools import setup, find_packages

setup(
    name="backtester",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.0",
        "numpy>=1.24",
        "requests>=2.31",
        "pyarrow>=14.0",
    ],
    entry_points={
        "console_scripts": [
            "backtester=backtester.cli:main",
        ],
    },
)
