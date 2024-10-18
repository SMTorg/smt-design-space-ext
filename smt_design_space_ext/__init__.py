from .version import __version__

# Symbols imported in smt to handle hierarchical variables
from smt_design_space_ext.design_space import (
    DesignSpace,
    ensure_design_space,
)

__all__ = ["__version__", "DesignSpace", "ensure_design_space"]
