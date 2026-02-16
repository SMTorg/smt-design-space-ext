from smt.design_space import (
    BaseDesignSpace,
    CategoricalVariable,
    DesignSpace,
    DesignVariable,
    FloatVariable,
    IntegerVariable,
    OrdinalVariable,
)

# Symbols imported in smt to handle hierarchical variables
from smt_design_space_ext.design_space import (
    HAS_ADSG,
    HAS_CONFIG_SPACE,
    FixedIntegerParam,
    NoDefaultConfigurationSpace,
)

from .version import __version__

if HAS_CONFIG_SPACE:
    from smt_design_space_ext.cs_ds_imp import ConfigSpaceDesignSpaceImpl
if HAS_ADSG:
    from smt_design_space_ext.adsg_ds_imp import AdsgDesignSpaceImpl

__all__ = [
    "__version__",
    "DesignSpace",
    "FloatVariable",
    "IntegerVariable",
    "OrdinalVariable",
    "CategoricalVariable",
    "BaseDesignSpace",
    "DesignVariable",
    "NoDefaultConfigurationSpace",
    "FixedIntegerParam",
    "HAS_CONFIG_SPACE",
    "HAS_ADSG",
    "ConfigSpaceDesignSpaceImpl",
    "AdsgDesignSpaceImpl",
]
