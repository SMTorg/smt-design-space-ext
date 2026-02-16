"""
Author: Jasper Bussemaker <jasper.bussemaker@dlr.de>

This package is distributed under New BSD license.
"""

import importlib
from typing import List, Union

import numpy as np

# Here we import design space base classes from smt
# We do not import smt.design_space as it would be circular!!!
from smt.design_space import (
    BaseDesignSpace,
    DesignSpace,
)

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
        center = self.to_value(value)
        lower, upper = self.lower, self.upper
        if upper - lower - 1 < number:
            neighbors = sorted(set(range(lower, upper + 1)) - {center})
            if transform:
                return neighbors
            return self.to_vector(np.asarray(neighbors)).tolist()

        return list(self.neighbors_vectorized(value, number, std=std, seed=rs))

    def neighbors_vectorized(self, vector, n, *, std=None, seed=None):
        """Sample neighbors without ConfigSpace 1.x strict vector legality check.

        ConfigSpace 1.x uses (x - lower) / (upper - lower) normalization while
        this project uses a custom normalization. The base class rejects our
        vectors as illegal, so we override to handle both formats.
        """
        center = self.to_value(vector)
        lower, upper = self.lower, self.upper
        all_neighbors = sorted(set(range(lower, upper + 1)) - {center})

        if len(all_neighbors) == 0:
            return np.array([], dtype=np.float64)

        if len(all_neighbors) <= n:
            return self.to_vector(np.asarray(all_neighbors))

        rng = seed if seed is not None else np.random
        chosen_idx = rng.choice(len(all_neighbors), size=n, replace=False)
        chosen = [all_neighbors[int(i)] for i in chosen_idx]
        return self.to_vector(np.asarray(chosen))
