"""
Author: Jasper Bussemaker <jasper.bussemaker@dlr.de>

This package is distributed under New BSD license.
"""

from typing import List, Union

import numpy as np


# Here we import design space base classes from smt
# We do not import smt.design_space as it would be circular!!!
from smt.design_space import (
    BaseDesignSpace,
    DesignSpace,
)
import importlib

spec_cs = importlib.util.find_spec("ConfigSpace")
if spec_cs:
    HAS_CONFIG_SPACE = True
    from ConfigSpace import (
        Configuration,
        ConfigurationSpace,
        UniformIntegerHyperparameter,
    )
else:
    HAS_CONFIG_SPACE = False

    class Configuration:
        pass

    class ConfigurationSpace:
        pass

    class UniformIntegerHyperparameter:
        pass


spec_adsg = importlib.util.find_spec("adsg_core")
if spec_adsg:
    HAS_ADSG = True
    from adsg_core import ADSG
else:
    HAS_ADSG = False


def ensure_design_space(xt=None, xlimits=None, design_space=None) -> "BaseDesignSpace":
    """Interface to turn legacy input formats into a DesignSpace"""

    if design_space is not None and isinstance(design_space, BaseDesignSpace):
        return design_space
    if HAS_ADSG and design_space is not None and isinstance(design_space, ADSG):
        return ValueError("Use AdsgDesignSpaceImpl instead.")

    if xlimits is not None:
        return ValueError("Use ConfigSpaceDesignSpaceImpl instead.")

    if xt is not None:
        return DesignSpace([[np.min(xt) - 0.99, np.max(xt) + 1e-4]] * xt.shape[1])

    raise ValueError("Nothing defined that could be interpreted as a design space!")


VarValueType = Union[int, str, List[Union[int, str]]]


class NoDefaultConfigurationSpace(ConfigurationSpace):
    """ConfigurationSpace that supports no default configuration"""

    def get_default_configuration(self, *args, **kwargs):
        raise NotImplementedError

    def _check_default_configuration(self, *args, **kwargs):
        pass


class FixedIntegerParam(UniformIntegerHyperparameter):
    def get_neighbors(
        self,
        value: float,
        rs: np.random.RandomState,
        number: int = 4,
        transform: bool = False,
        std: float = 0.2,
    ) -> List[int]:
        # Temporary fix until https://github.com/automl/ConfigSpace/pull/313 is released
        center = self._transform(value)
        lower, upper = self.lower, self.upper
        if upper - lower - 1 < number:
            neighbors = sorted(set(range(lower, upper + 1)) - {center})
            if transform:
                return neighbors
            return self._inverse_transform(np.asarray(neighbors)).tolist()

        return super().get_neighbors(
            value, rs, number=number, transform=transform, std=std
        )
