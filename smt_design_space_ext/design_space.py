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

try:
    from ConfigSpace import (
        CategoricalHyperparameter,
        Configuration,
        ConfigurationSpace,
        EqualsCondition,
        ForbiddenAndConjunction,
        ForbiddenEqualsClause,
        ForbiddenInClause,
        ForbiddenLessThanRelation,
        InCondition,
        OrdinalHyperparameter,
        UniformFloatHyperparameter,
        UniformIntegerHyperparameter,
    )
    from ConfigSpace.exceptions import ForbiddenValueError
    from ConfigSpace.util import get_random_neighbor

    HAS_CONFIG_SPACE = True

except ImportError:
    HAS_CONFIG_SPACE = False
try:
    from adsg_core.graph.graph_edges import EdgeType
    from adsg_core import GraphProcessor, SelectionChoiceNode
    from adsg_core.graph.adsg import ADSG
    from adsg_core import BasicADSG, NamedNode, DesignVariableNode

    HAS_ADSG = True
except ImportError:
    HAS_ADSG = False

    class Configuration:
        pass

    class ConfigurationSpace:
        pass

    class UniformIntegerHyperparameter:
        pass


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
