"""
Author: Paul Saves <paul.saves@onera.fr>
        Remi Lafage <remi.lafage@onera.fr>
        Jasper Bussemaker <jasper.bussemaker@dlr.de>

This package is distributed under New BSD license.
"""

from setuptools import setup

# Import __version__ without importing the module in setup
exec(open("./smt_design_space_ext/version.py").read())

setup(
    name="smt_design_space_ext",
    version=__version__,  # noqa
    author="Paul Saves et al.",
    author_email="paul.saves@onera.fr",
    keywords=["SMT, DesignSpace, Graph"],
    license="BSD-3",
    description="SMT design space extension for hierarchical variables handling",
    install_requires=[
        "numpy",
        "smt>=2.10",
        "ConfigSpace>=1.2.0",
        "adsg-core==1.2.0",
    ],
    maintainer="Paul Saves",
    maintainer_email="paul.saves@onera.fr",
    packages=[
        "smt_design_space_ext",
    ],
    python_requires=">=3.9",
    url="https://github.com/SMTorg/smt-design-space",  # use the URL to the github repo
    download_url="https://github.com/SMTorg/smt-design-space/releases",
)
