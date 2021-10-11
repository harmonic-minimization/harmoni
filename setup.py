# Copyright (C) 2021 Mina Jamshidi
# <minajamshidi91@gmail.com>

from setuptools import setup

with open('README.md', "r") as fh:
    long_description = fh.read()

setup(
    author='Mina Jamshidi',
    author_email='minajamshidi91@gmail.com',
    url='https://github.com/harmonic-minimization/harmoni',
    name='harmoni',
    version='0.0.5',
    description='harmonic minimization method for eliminating harmonic-driven connectivity with MEEG',
    long_description=long_description,
    long_description_content_type="text/markdown",
    py_modules=["harmoni/extratools", "harmoni/harmonitools"],
    package_dir={'': 'src'},
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy>=1.10",
        "scipy>=0.19",
    ],
    extras_require={
        "dev": [
                "pytest>=6.2",
                "pytest-cov>=2.12",
                "check-manifest>=0.40",
                "flake8>=3.9.2",
                "mypy>=0.910",
                "tox>=3.24",
        ],
    },
)

