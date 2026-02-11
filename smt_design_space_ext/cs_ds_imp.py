#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 15:50:49 2024

@author: psaves
"""

from typing import List, Sequence, Tuple, Union

import numpy as np

from smt.sampling_methods import LHS

# Here we import design space base classes from smt
# We do not import smt.design_space as it would be circular!!!
from smt_design_space_ext import (
    FloatVariable,
    IntegerVariable,
    OrdinalVariable,
    CategoricalVariable,
    BaseDesignSpace,
    DesignVariable,
    HAS_ADSG,
    HAS_CONFIG_SPACE,
)

from ConfigSpace import (
    CategoricalHyperparameter,
    Configuration,
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

if HAS_ADSG:
    from adsg_core.graph.adsg import ADSG


from smt_design_space_ext import (
    FixedIntegerParam,
    NoDefaultConfigurationSpace,
)


VarValueType = Union[int, str, List[Union[int, str]]]


def ensure_design_space(xt=None, xlimits=None, design_space=None) -> "BaseDesignSpace":
    """Interface to turn legacy input formats into a DesignSpace"""

    if design_space is not None and isinstance(design_space, BaseDesignSpace):
        return design_space
    if HAS_ADSG and design_space is not None and isinstance(design_space, ADSG):
        return ValueError("Use AdsgDesignSpaceImpl instead.")

    if xlimits is not None:
        return ConfigSpaceDesignSpaceImpl(xlimits)

    if xt is not None:
        return ConfigSpaceDesignSpaceImpl(
            [[np.min(xt) - 0.99, np.max(xt) + 1e-4]] * xt.shape[1]
        )

    raise ValueError("Nothing defined that could be interpreted as a design space!")


class ConfigSpaceDesignSpaceImpl(BaseDesignSpace):
    """
    Class for defining a (hierarchical) design space by defining design variables, defining decreed variables
    (optional), and adding value constraints (optional).

    Numerical bounds can be requested using `get_num_bounds()`.
    If needed, it is possible to get the legacy SMT < 2.0 `xlimits` format using `get_x_limits()`.

    Parameters
    ----------
    design_variables: list[DesignVariable]
       - The list of design variables: FloatVariable, IntegerVariable, OrdinalVariable, or CategoricalVariable

    Examples
    --------
    Instantiate the design space with all its design variables:

    >>> from smt.utils.design_space import *
    >>> ds = DesignSpace([
    >>>     CategoricalVariable(['A', 'B']),  # x0 categorical: A or B; order is not relevant
    >>>     OrdinalVariable(['C', 'D', 'E']),  # x1 ordinal: C, D or E; order is relevant
    >>>     IntegerVariable(0, 2),  # x2 integer between 0 and 2 (inclusive): 0, 1, 2
    >>>     FloatVariable(0, 1),  # c3 continuous between 0 and 1
    >>> ])
    >>> assert len(ds.design_variables) == 4

    You can define decreed variables (conditional activation):

    >>> ds.declare_decreed_var(decreed_var=1, meta_var=0, meta_value='A')  # Activate x1 if x0 == A

    Decreed variables can be chained (however no cycles and no "diamonds" are supported):
    Note: only if ConfigSpace is installed! pip install smt[cs]
    >>> ds.declare_decreed_var(decreed_var=2, meta_var=1, meta_value=['C', 'D'])  # Activate x2 if x1 == C or D

    If combinations of values between two variables are not allowed, this can be done using a value constraint:
    Note: only if ConfigSpace is installed! pip install smt[cs]
    >>> ds.add_value_constraint(var1=0, value1='A', var2=2, value2=[0, 1])  # Forbid x0 == A && x2 == 0 or 1

    After defining everything correctly, you can then use the design space object to correct design vectors and get
    information about which design variables are acting:

    >>> x_corr, is_acting = ds.correct_get_acting(np.array([
    >>>     [0, 0, 2, .25],
    >>>     [0, 2, 1, .75],
    >>> ]))
    >>> assert np.all(x_corr == np.array([
    >>>     [0, 0, 2, .25],
    >>>     [0, 2, 0, .75],
    >>> ]))
    >>> assert np.all(is_acting == np.array([
    >>>     [True, True, True, True],
    >>>     [True, True, False, True],  # x2 is not acting if x1 != C or D (0 or 1)
    >>> ]))

    It is also possible to randomly sample design vectors conforming to the constraints:

    >>> x_sampled, is_acting_sampled = ds.sample_valid_x(100)

    You can also instantiate a purely-continuous design space from bounds directly:

    >>> continuous_design_space = DesignSpace([(0, 1), (0, 2), (.5, 5.5)])
    >>> assert continuous_design_space.n_dv == 3

    If needed, it is possible to get the legacy design space definition format:

    >>> xlimits = ds.get_x_limits()
    >>> cont_bounds = ds.get_num_bounds()
    >>> unfolded_cont_bounds = ds.get_unfolded_num_bounds()

    """

    def __init__(
        self,
        design_variables: Union[List[DesignVariable], list, np.ndarray],
        seed=None,
    ):
        self.sampler = None

        # Assume float variable bounds as inputs
        def _is_num(val):
            try:
                float(val)
                return True
            except ValueError:
                return False

        if len(design_variables) > 0 and not isinstance(
            design_variables[0], DesignVariable
        ):
            converted_dvs = []
            for bounds in design_variables:
                if len(bounds) != 2 or not _is_num(bounds[0]) or not _is_num(bounds[1]):
                    raise RuntimeError(
                        f"Expecting either a list of DesignVariable objects or float variable "
                        f"bounds! Unrecognized: {bounds!r}"
                    )
                converted_dvs.append(FloatVariable(bounds[0], bounds[1]))
            design_variables = converted_dvs

        self._cs = None
        self._cs_cate = None
        if HAS_CONFIG_SPACE:
            cs_vars = {}
            cs_vars_cate = {}
            self.isinteger = False
            for i, dv in enumerate(design_variables):
                name = f"x{i}"
                if isinstance(dv, FloatVariable):
                    cs_vars[name] = UniformFloatHyperparameter(
                        name, lower=dv.lower, upper=dv.upper
                    )
                    cs_vars_cate[name] = UniformFloatHyperparameter(
                        name, lower=dv.lower, upper=dv.upper
                    )
                elif isinstance(dv, IntegerVariable):
                    cs_vars[name] = FixedIntegerParam(
                        name, lower=dv.lower, upper=dv.upper
                    )
                    listvalues = []
                    for i in range(int(dv.upper - dv.lower + 1)):
                        listvalues.append(str(int(i + dv.lower)))
                    cs_vars_cate[name] = CategoricalHyperparameter(
                        name, choices=listvalues
                    )
                    self.isinteger = True
                elif isinstance(dv, OrdinalVariable):
                    cs_vars[name] = OrdinalHyperparameter(name, sequence=dv.values)
                    cs_vars_cate[name] = CategoricalHyperparameter(
                        name, choices=dv.values
                    )

                elif isinstance(dv, CategoricalVariable):
                    cs_vars[name] = CategoricalHyperparameter(name, choices=dv.values)
                    cs_vars_cate[name] = CategoricalHyperparameter(
                        name, choices=dv.values
                    )

                else:
                    raise ValueError(f"Unknown variable type: {dv!r}")

            cs_seed = seed if isinstance(seed, (int, np.integer)) else None
            self._cs = NoDefaultConfigurationSpace(space=cs_vars, seed=cs_seed)
            ## Fix to make constraints work correctly with either IntegerVariable or OrdinalVariable
            ## ConfigSpace is malfunctioning
            self._cs_cate = NoDefaultConfigurationSpace(
                space=cs_vars_cate, seed=cs_seed
            )

        # dict[int, dict[any, list[int]]]: {meta_var_idx: {value: [decreed_var_idx, ...], ...}, ...}
        self._meta_vars = {}
        self._is_decreed = np.zeros((len(design_variables),), dtype=bool)

        super().__init__(design_variables=design_variables, seed=seed)

    def declare_decreed_var(
        self, decreed_var: int, meta_var: int, meta_value: VarValueType
    ):
        """
        Define a conditional (decreed) variable to be active when the meta variable has (one of) the provided values.

        Parameters
        ----------
        decreed_var: int
           - Index of the conditional variable (the variable that is conditionally active)
        meta_var: int
           - Index of the meta variable (the variable that determines whether the conditional var is active)
        meta_value: int | str | list[int|str]
           - The value or list of values that the meta variable can have to activate the decreed var
        """

        # ConfigSpace implementation
        if self._cs is not None:
            # Get associated parameters
            decreed_param = self._get_param(decreed_var)
            meta_param = self._get_param(meta_var)

            # Add a condition that checks for equality (if single value given) or in-collection (if sequence given)
            if isinstance(meta_value, Sequence):
                condition = InCondition(decreed_param, meta_param, meta_value)
            else:
                condition = EqualsCondition(decreed_param, meta_param, meta_value)

            ## Fix to make constraints work correctly with either IntegerVariable or OrdinalVariable
            ## ConfigSpace is malfunctioning
            self._cs.add(condition)
            decreed_param = self._get_param2(decreed_var)
            meta_param = self._get_param2(meta_var)
            # Add a condition that checks for equality (if single value given) or in-collection (if sequence given)
            if isinstance(meta_value, Sequence):
                try:
                    condition = InCondition(
                        decreed_param,
                        meta_param,
                        list(np.atleast_1d(np.array(meta_value, dtype=str))),
                    )
                except ValueError:
                    condition = InCondition(
                        decreed_param,
                        meta_param,
                        list(np.atleast_1d(np.array(meta_value, dtype=float))),
                    )
            else:
                try:
                    condition = EqualsCondition(
                        decreed_param, meta_param, str(meta_value)
                    )
                except ValueError:
                    condition = EqualsCondition(decreed_param, meta_param, meta_value)

            self._cs_cate.add(condition)

        # Simplified implementation
        else:
            # Variables cannot be both meta and decreed at the same time
            if self._is_decreed[meta_var]:
                raise RuntimeError(
                    f"Variable cannot be both meta and decreed ({meta_var})!"
                )

            # Variables can only be decreed by one meta var
            if self._is_decreed[decreed_var]:
                raise RuntimeError(f"Variable is already decreed: {decreed_var}")

            # Define meta-decreed relationship
            if meta_var not in self._meta_vars:
                self._meta_vars[meta_var] = {}

            meta_var_obj = self.design_variables[meta_var]
            for value in (
                meta_value if isinstance(meta_value, Sequence) else [meta_value]
            ):
                encoded_value = value
                if isinstance(meta_var_obj, (OrdinalVariable, CategoricalVariable)):
                    if value in meta_var_obj.values:
                        encoded_value = meta_var_obj.values.index(value)

                if encoded_value not in self._meta_vars[meta_var]:
                    self._meta_vars[meta_var][encoded_value] = []
                self._meta_vars[meta_var][encoded_value].append(decreed_var)

        # Mark as decreed (conditionally acting)
        self._is_decreed[decreed_var] = True

    def add_value_constraint(
        self, var1: int, value1: VarValueType, var2: int, value2: VarValueType
    ):
        """
        Define a constraint where two variables cannot have the given values at the same time.

        Parameters
        ----------
        var1: int
           - Index of the first variable
        value1: int | str | list[int|str]
           - Value or values that the first variable is checked against
        var2: int
           - Index of the second variable
        value2: int | str | list[int|str]
           - Value or values that the second variable is checked against
        """
        # Get parameters
        param1 = self._get_param(var1)
        param2 = self._get_param(var2)
        mixint_types = (UniformIntegerHyperparameter, OrdinalHyperparameter)
        self.has_valcons_ord_int = isinstance(param1, mixint_types) or isinstance(
            param2, mixint_types
        )
        if not (isinstance(param1, UniformFloatHyperparameter)) and not (
            isinstance(param2, UniformFloatHyperparameter)
        ):
            # Add forbidden clauses
            if isinstance(value1, Sequence):
                clause1 = ForbiddenInClause(param1, value1)
            else:
                clause1 = ForbiddenEqualsClause(param1, value1)

            if isinstance(value2, Sequence):
                clause2 = ForbiddenInClause(param2, value2)
            else:
                clause2 = ForbiddenEqualsClause(param2, value2)

            constraint_clause = ForbiddenAndConjunction(clause1, clause2)
            self._cs.add(constraint_clause)
        else:
            if value1 in [">", "<"] and value2 in [">", "<"] and value1 != value2:
                if value1 == "<":
                    constraint_clause = ForbiddenLessThanRelation(param1, param2)
                    self._cs.add(constraint_clause)
                else:
                    constraint_clause = ForbiddenLessThanRelation(param2, param1)
                    self._cs.add(constraint_clause)
            else:
                raise ValueError("Bad definition of DesignSpace.")

        ## Fix to make constraints work correctly with either IntegerVariable or OrdinalVariable
        ## ConfigSpace is malfunctioning
        # Get parameters
        param1 = self._get_param2(var1)
        param2 = self._get_param2(var2)
        # Add forbidden clauses
        if not (isinstance(param1, UniformFloatHyperparameter)) and not (
            isinstance(param2, UniformFloatHyperparameter)
        ):
            if isinstance(value1, Sequence):
                clause1 = ForbiddenInClause(
                    param1, list(np.atleast_1d(np.array(value1, dtype=str)))
                )
            else:
                clause1 = ForbiddenEqualsClause(param1, str(value1))

            if isinstance(value2, Sequence):
                try:
                    clause2 = ForbiddenInClause(
                        param2, list(np.atleast_1d(np.array(value2, dtype=str)))
                    )
                except ValueError:
                    clause2 = ForbiddenInClause(
                        param2, list(np.atleast_1d(np.array(value2, dtype=float)))
                    )
            else:
                try:
                    clause2 = ForbiddenEqualsClause(param2, str(value2))
                except ValueError:
                    clause2 = ForbiddenEqualsClause(param2, value2)

            constraint_clause = ForbiddenAndConjunction(clause1, clause2)
            self._cs_cate.add(constraint_clause)

    def _get_param(self, idx):
        try:
            return self._cs[f"x{idx}"]
        except KeyError:
            raise KeyError(f"Variable not found: {idx}")

    def _get_param2(self, idx):
        try:
            return self._cs_cate[f"x{idx}"]
        except KeyError:
            raise KeyError(f"Variable not found: {idx}")

    @property
    def _cs_var_idx(self):
        """
        ConfigurationSpace applies topological sort when adding conditions, so compared to what we expect the order of
        parameters might have changed.

        This property contains the indices of the params in the ConfigurationSpace.
        """
        names = list(self._cs.keys())
        return np.array(
            [names.index(f"x{ix}") for ix in range(len(self.design_variables))]
        )

    @property
    def _inv_cs_var_idx(self):
        """
        See _cs_var_idx. This function returns the opposite mapping: the positions of our design variables for each
        param.
        """
        return np.array([int(param[1:]) for param in self._cs.keys()])

    def _is_conditionally_acting(self) -> np.ndarray:
        # Decreed variables are the conditionally acting variables
        return self._is_decreed

    def _correct_get_acting(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Correct and impute design vectors"""
        x = x.astype(float)
        if self._cs is not None:
            # Normalize value according to what ConfigSpace expects
            self._normalize_x(x)

            # Get corrected Configuration objects by mapping our design vectors
            # to the ordering of the ConfigurationSpace
            inv_cs_var_idx = self._inv_cs_var_idx
            configs = []
            for xi in x:
                configs.append(self._get_correct_config(xi[inv_cs_var_idx]))

            # Convert Configuration objects to design vectors and get the is_active matrix
            x_out, is_act = self._configs_to_x(configs)
            self._impute_non_acting(x_out, is_act)
            return x_out, is_act

        # Simplified implementation
        # Correct discrete variables
        x_corr = x.copy()
        self._normalize_x(x_corr, cs_normalize=False)

        # Determine which variables are acting
        is_acting = np.ones(x_corr.shape, dtype=bool)
        is_acting[:, self._is_decreed] = False
        for i, xi in enumerate(x_corr):
            for i_meta, decrees in self._meta_vars.items():
                meta_var_value = xi[i_meta]
                if meta_var_value in decrees:
                    i_decreed_vars = decrees[meta_var_value]
                    is_acting[i, i_decreed_vars] = True

        # Impute non-acting variables
        self._impute_non_acting(x_corr, is_acting)

        return x_corr, is_acting

    def _get_int_seed(self, vector=None):
        """Convert self.seed to an int for ConfigSpace 1.x API calls.
        If vector is provided, derive a deterministic seed from it for
        reproducible forbidden-value neighbor generation.
        """
        if vector is not None:
            # Deterministic seed based on vector content
            finite_vals = vector[np.isfinite(vector)]
            return abs(hash(tuple(finite_vals.tolist()))) % (2**31)
        if isinstance(self.seed, np.random.Generator):
            return int(self.seed.integers(0, 2**31))
        elif self.seed is not None:
            return int(self.seed)
        return 0

    def _sample_valid_x(self, n: int, seed=None) -> Tuple[np.ndarray, np.ndarray]:
        """Sample design vectors"""
        # Simplified implementation: sample design vectors in unfolded space
        x_limits_unfolded = self.get_unfolded_num_bounds()

        if self._cs is not None:
            # Sample Configuration objects
            # ConfigSpace 1.x seed() expects an int, but smt base class
            # sets self.seed to a np.random.Generator object
            cs_seed = self._get_int_seed()
            self._cs.seed(cs_seed)
            configs = self._cs.sample_configuration(n)
            if n == 1:
                configs = [configs]
            # Convert Configuration objects to design vectors and get the is_active matrix
            return self._configs_to_x(configs)

        else:
            if self.sampler is None:
                self.sampler = LHS(
                    xlimits=x_limits_unfolded,
                    seed=seed if seed is not None else self.seed,
                    criterion="ese",
                )
            x = self.sampler(n)
            # Fold and cast to discrete
            x, _ = self.fold_x(x)
            self._normalize_x(x, cs_normalize=False)
            # Get acting information and impute
            return self.correct_get_acting(x)

    def _get_correct_config(self, vector: np.ndarray) -> Configuration:
        # Determine active hyperparameters and set inactive ones to NaN
        # This replaces the old workaround of catching check_valid_configuration errors
        # https://github.com/automl/ConfigSpace/issues/253#issuecomment-1513216665
        all_hp_names = list(self._cs.keys())
        while True:
            active_hps = self._cs.get_active_hyperparameters(vector)
            changed = False
            for i, name in enumerate(all_hp_names):
                if name not in active_hps and not np.isnan(vector[i]):
                    vector[i] = np.nan
                    changed = True
            if not changed:
                break

        config = Configuration(self._cs, vector=vector)

        ## Fix to make constraints work correctly with either IntegerVariable or OrdinalVariable
        ## ConfigSpace is malfunctioning
        if self.isinteger and self.has_valcons_ord_int:
            vector2 = np.copy(vector)
            self._cs_denormalize_x_ordered(np.atleast_2d(vector2))
            indvec = 0
            for hp in self._cs_cate:
                if (
                    (str(self._cs[hp]).split()[2]) == "UniformInteger,"
                    and (str(self._cs_cate[hp]).split()[2][:3]) == "Cat"
                    and not (np.isnan(vector2[indvec]))
                ):
                    vector2[indvec] = int(vector2[indvec]) - int(
                        str(self._cs_cate[hp]).split()[4][1:-1]
                    )
                indvec += 1
            self._normalize_x_no_integer(np.atleast_2d(vector2))

            try:
                self._cs_cate._check_forbidden(vector2)
            except ForbiddenValueError:
                vector = config.get_array().copy()
                indvec = 0
                vector2 = np.copy(vector)
                for hp in self._cs_cate:
                    if (str(self._cs_cate[hp]).split()[2][:3]) == "Cat" and not (
                        np.isnan(vector2[indvec])
                    ):
                        vector2[indvec] = int(vector2[indvec])
                    indvec += 1

                config2 = Configuration(self._cs_cate, vector=vector2)
                int_seed = self._get_int_seed(vector=vector)
                config3 = get_random_neighbor(config2, seed=int_seed)
                # Convert _cs_cate values to _cs values (categorical indices
                # differ from integer HP vector normalization)
                new_values = {}
                for hp_name in dict(config3):
                    hp_cs = self._cs[hp_name]
                    value = config3[hp_name]
                    if isinstance(hp_cs, UniformIntegerHyperparameter):
                        new_values[hp_name] = int(value)
                    else:
                        new_values[hp_name] = value
                config4 = Configuration(self._cs, values=new_values)
                return config4

        # Check forbidden clauses on the main config
        try:
            self._cs._check_forbidden(config.get_array())
        except ForbiddenValueError:
            int_seed = self._get_int_seed(vector=config.get_array())
            if not (self.has_valcons_ord_int):
                return get_random_neighbor(config, seed=int_seed)
            else:
                return get_random_neighbor(config, seed=int_seed)

        return config

    def _configs_to_x(
        self, configs: List["Configuration"]
    ) -> Tuple[np.ndarray, np.ndarray]:
        x = np.zeros((len(configs), len(self.design_variables)))
        is_acting = np.zeros(x.shape, dtype=bool)
        if len(configs) == 0:
            return x, is_acting

        cs_var_idx = self._cs_var_idx
        for i, config in enumerate(configs):
            x[i, :] = config.get_array()[cs_var_idx]

        # De-normalize continuous and integer variables
        self._cs_denormalize_x(x)

        # Set is_active flags and impute x
        is_acting = np.isfinite(x)
        self._impute_non_acting(x, is_acting)

        return x, is_acting

    def _impute_non_acting(self, x: np.ndarray, is_acting: np.ndarray):
        for i, dv in enumerate(self.design_variables):
            if isinstance(dv, FloatVariable):
                # Impute continuous variables to the mid of their bounds
                x[~is_acting[:, i], i] = 0.5 * (dv.upper - dv.lower) + dv.lower

            else:
                # Impute discrete variables to their lower bounds
                lower = 0
                if isinstance(dv, (IntegerVariable, OrdinalVariable)):
                    lower = dv.lower

                x[~is_acting[:, i], i] = lower

    def _normalize_x(self, x: np.ndarray, cs_normalize=True):
        for i, dv in enumerate(self.design_variables):
            if isinstance(dv, FloatVariable):
                if cs_normalize:
                    dv.lower = min(np.min(x[:, i]), dv.lower)
                    dv.upper = max(np.max(x[:, i]), dv.upper)
                    x[:, i] = np.clip(
                        (x[:, i] - dv.lower) / (dv.upper - dv.lower + 1e-16), 0, 1
                    )

            elif isinstance(dv, IntegerVariable):
                x[:, i] = self._round_equally_distributed(x[:, i], dv.lower, dv.upper)

                if cs_normalize:
                    # Normalize between 0 and 1, matching ConfigSpace 1.x normalization
                    x[:, i] = (x[:, i] - dv.lower) / max(dv.upper - dv.lower, 1)

    def _normalize_x_no_integer(self, x: np.ndarray, cs_normalize=True):
        ordereddesign_variables = [
            self.design_variables[i] for i in self._inv_cs_var_idx
        ]
        for i, dv in enumerate(ordereddesign_variables):
            if isinstance(dv, FloatVariable):
                if cs_normalize:
                    x[:, i] = np.clip(
                        (x[:, i] - dv.lower) / (dv.upper - dv.lower + 1e-16), 0, 1
                    )

            elif isinstance(dv, (OrdinalVariable, CategoricalVariable)):
                # To ensure equal distribution of continuous values to discrete values, we first stretch-out the
                # continuous values to extend to 0.5 beyond the integer limits and then round. This ensures that the
                # values at the limits get a large-enough share of the continuous values
                x[:, i] = self._round_equally_distributed(x[:, i], dv.lower, dv.upper)

    def _cs_denormalize_x(self, x: np.ndarray):
        for i, dv in enumerate(self.design_variables):
            if isinstance(dv, FloatVariable):
                x[:, i] = x[:, i] * (dv.upper - dv.lower) + dv.lower

            elif isinstance(dv, IntegerVariable):
                # Denormalize matching ConfigSpace 1.x normalization
                x[:, i] = np.round(x[:, i] * (dv.upper - dv.lower) + dv.lower)

    def _cs_denormalize_x_ordered(self, x: np.ndarray):
        ordereddesign_variables = [
            self.design_variables[i] for i in self._inv_cs_var_idx
        ]
        for i, dv in enumerate(ordereddesign_variables):
            if isinstance(dv, FloatVariable):
                x[:, i] = x[:, i] * (dv.upper - dv.lower) + dv.lower

            elif isinstance(dv, IntegerVariable):
                # Denormalize matching ConfigSpace 1.x normalization
                x[:, i] = np.round(x[:, i] * (dv.upper - dv.lower) + dv.lower)

    def __str__(self):
        dvs = "\n".join([f"x{i}: {dv!s}" for i, dv in enumerate(self.design_variables)])
        return f"Design space:\n{dvs}"

    def __repr__(self):
        return f"{self.__class__.__name__}({self.design_variables!r})"
