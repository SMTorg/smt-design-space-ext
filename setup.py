"""
Author: Paul Saves <paul.saves@onera.fr>
        Remi Lafage <remi.lafage@onera.fr>
        Jasper Bussemaker <jasper.bussemaker@dlr.de>

This package is distributed under New BSD license.
"""

import sys

from setuptools import setup

version = {}
with open("./__version__.py") as fp:
    exec(fp.read(), version)



setup(
    name="SMTDesignSpace",
    author="Paul Saves et al.",
    author_email="paul.saves@onera.fr",
    keywords=["SMT, DesignSpace, Graph"],
    license="ONERA",
    description="SMT Design Space",
    install_requires=[
        "scipy>=1.11.3",
        "moe>=3.0.0",
        "numpy>=1.23.5",
        "scikit-learn==1.4.0",
        "smt>=2.3.0",
        "nlopt>=2.6.2",
        "six>=1.16.0",
        "cvxopt>=1.2.0",
        "ConfigSpace==0.6.1",
        "adsg-core==1.1.0",
    ],
    maintainer="Paul Saves",
    maintainer_email="paul.saves@onera.fr",
    packages=[
        "SMTDesignSpace",
    ],
    include_package_data=True,
    python_requires=">=3.8",
    version=version["__version__"],
    zip_safe=False,
    url="https://github.com/SMTorg/smt-design-space",  # use the URL to the github repo
    download_url="https://github.com/SMTorg/smt-design-space/releases",
)
