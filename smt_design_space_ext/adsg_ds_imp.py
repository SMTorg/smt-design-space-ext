#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 15:50:49 2024

@author: psaves
"""

from typing import List, Tuple, Union

import numpy as np


# Here we import design space base classes from smt
# We do not import smt.design_space as it would be circular!!!
from smt_design_space_ext import (
    FloatVariable,
    IntegerVariable,
    OrdinalVariable,
    CategoricalVariable,
    BaseDesignSpace,
    ConfigSpaceDesignSpaceImpl,
    HAS_ADSG,
    HAS_CONFIG_SPACE,
)


from adsg_core.graph.graph_edges import EdgeType
from adsg_core import GraphProcessor, SelectionChoiceNode
from adsg_core.graph.adsg import ADSG
from adsg_core import BasicADSG, NamedNode, DesignVariableNode


VarValueType = Union[int, str, List[Union[int, str]]]


def ensure_design_space(xt=None, xlimits=None, design_space=None) -> "BaseDesignSpace":
    """Interface to turn legacy input formats into a DesignSpace"""

    if design_space is not None and isinstance(design_space, BaseDesignSpace):
        return design_space
    if HAS_ADSG and design_space is not None and isinstance(design_space, ADSG):
        return _convert_adsg_to_legacy(design_space)

    if xlimits is not None:
        return ConfigSpaceDesignSpaceImpl(xlimits)

    if xt is not None:
        return ConfigSpaceDesignSpaceImpl(
            [[np.min(xt) - 0.99, np.max(xt) + 1e-4]] * xt.shape[1]
        )

    raise ValueError("Nothing defined that could be interpreted as a design space!")


def _convert_adsg_to_legacy(adsg) -> "BaseDesignSpace":
    """Interface to turn adsg input formats into legacy DesignSpace"""
    gp = GraphProcessor(adsg)
    listvar = []
    gvars = gp._all_des_var_data[0]
    varnames = [ii.name for ii in gvars]
    for i in gvars:
        if i._bounds is not None:
            listvar.append(FloatVariable(lower=i._bounds[0], upper=i._bounds[1]))
        elif type(i.node) is SelectionChoiceNode:
            a = (
                str(i._opts)
                .replace("[", "")
                .replace("]", "")
                .replace(" ", "")
                .replace("'", "")
                .split(",")
            )
            listvar.append(CategoricalVariable(a))
        else:
            a = (
                str(i._opts)
                .replace("[", "")
                .replace("]", "")
                .replace(" ", "")
                .replace("'", "")
                .split(",")
            )
            listvar.append(OrdinalVariable(a))

    design_space = ConfigSpaceDesignSpaceImpl(listvar)

    active_vars = [i for i, x in enumerate(gp.dv_is_conditionally_active) if x]
    nodelist = list(adsg._graph.nodes)
    nodenamelist = [
        element.strip()[1:-1]
        for element in str(list(adsg._graph.nodes))[1:-1]
        .replace("D[Sel:", "[")
        .replace("DV[", "[")
        .replace(" ", "")
        .split(",")
        if element.strip().startswith("[") and element.strip().endswith("]")
    ]
    for i in range(np.sum(gp.dv_is_conditionally_active)):
        meta_values = [
            metav
            for metav in iter(
                adsg._graph.predecessors(
                    nodelist[nodenamelist.index(gvars[active_vars[i]].name)]
                )
            )
        ]
        meta_variable = next(iter(adsg._graph.predecessors(meta_values[0])))
        while str(meta_variable).split("[")[0] != "D":
            meta_values = [
                metav for metav in iter((adsg._graph.predecessors(meta_values[0])))
            ]
            meta_variable = next(iter(adsg._graph.predecessors(meta_values[0])))
        namemetavar = (
            str(meta_variable)
            .replace("D[Sel:", "")
            .replace("DV[", "")
            .replace(" ", "")
            .replace("[", "")
            .replace("]", "")
        )
        design_space.declare_decreed_var(
            decreed_var=active_vars[i],
            meta_var=varnames.index(namemetavar),
            meta_value=[str(metaval)[1:-1] for metaval in meta_values],
        )

    edges = np.array(list(adsg._graph.edges.data()))
    if len(edges) > 0:
        edgestype = [edge["type"] for edge in edges[:, 2]]
        incomp_nodes = []
        for i, edge in enumerate(edges):
            if edgestype[i] == EdgeType.INCOMPATIBILITY:
                incomp_nodes.append([edges[i][0], edges[i][1]])

        def remove_symmetry(lst):
            unique_pairs = set()

            for pair in lst:
                # Sort the pair based on the _id attribute of NamedNode
                sorted_pair = tuple(sorted(pair, key=lambda node: node._id))
                unique_pairs.add(sorted_pair)

            # Convert set of tuples back to list of lists if needed
            return [list(pair) for pair in unique_pairs]

        incomp_nodes = remove_symmetry(incomp_nodes)

        for pair in incomp_nodes:
            node1, node2 = pair
            vars1 = next(iter(adsg._graph.predecessors(node1)))
            while str(vars1).split("[")[0] != "D":
                vars1 = next(iter(adsg._graph.predecessors(node1)))
            vars2 = next(iter(adsg._graph.predecessors(node2)))
            while str(vars1).split("[")[0] != "D":
                vars2 = next(iter(adsg._graph.predecessors(node2)))
        for pair in incomp_nodes:
            node1, node2 = pair
            vars1 = next(iter(adsg._graph.predecessors(node1)))
            while str(vars1).split("[")[0] != "D":
                vars1 = next(iter(adsg._graph.predecessors(node1)))
            vars2 = next(iter(adsg._graph.predecessors(node2)))
            while str(vars1).split("[")[0] != "D":
                vars2 = next(iter(adsg._graph.predecessors(node2)))
            namevar1 = (
                str(vars1)
                .replace("D[Sel:", "")
                .replace("DV[", "")
                .replace(" ", "")
                .replace("[", "")
                .replace("]", "")
            )
            namevar2 = (
                str(vars2)
                .replace("D[Sel:", "")
                .replace("DV[", "")
                .replace(" ", "")
                .replace("[", "")
                .replace("]", "")
            )
            design_space.add_value_constraint(
                var1=varnames.index(namevar1),
                value1=[str(node1)[1:-1]],
                var2=varnames.index(namevar2),
                value2=[str(node2)[1:-1]],
            )  # Forbid more than 35 neurons with ASGD

    return design_space


def _legacy_to_adsg(legacy_ds: "ConfigSpaceDesignSpaceImpl") -> "BasicADSG":
    """
    Interface to turn a legacy DesignSpace back into an ADSG instance.

    Parameters:
    legacy_ds (ConfigSpaceDesignSpaceImpl): The legacy ConfigSpaceDesignSpaceImpl instance.

    Returns:
    BasicADSG: The corresponding ADSG graph.
    """
    adsg = BasicADSG()

    # Create nodes for each variable in the DesignSpace
    nodes = {}
    value_nodes = {}  # This will store decreed value nodes
    start_nodes = set()
    for i, var in enumerate(legacy_ds._design_variables):
        if isinstance(var, FloatVariable):
            # Create a DesignVariableNode with bounds for continuous variables
            var_node = DesignVariableNode(f"x{i}", bounds=(var.lower, var.upper))
        elif isinstance(var, IntegerVariable):
            # Create a SelectionChoiceNode for ordinal variables (ordinal treated like categorical)
            var_node = NamedNode(f"x{i}")
            valuesord = list(range(var.lower, var.upper + 1))
            choices = [NamedNode(value) for value in valuesord]
            value_nodes[f"x{i}"] = (
                choices  # Store decreed value nodes for this variable
            )
            adsg.add_selection_choice(f"choice_x{i}", var_node, choices)
        elif isinstance(var, CategoricalVariable):
            # Create a SelectionChoiceNode for categorical variables
            var_node = NamedNode(f"x{i}")
            choices = [NamedNode(value) for value in var.values]
            value_nodes[f"x{i}"] = (
                choices  # Store decreed value nodes for this variable
            )
            adsg.add_selection_choice(f"choice_x{i}", var_node, choices)
        elif isinstance(var, OrdinalVariable):
            # Create a SelectionChoiceNode for ordinal variables (ordinal treated like categorical)
            var_node = NamedNode(f"x{i}")
            choices = [NamedNode(value) for value in var.values]
            value_nodes[f"x{i}"] = (
                choices  # Store decreed value nodes for this variable
            )
            adsg.add_selection_choice(f"choice_x{i}", var_node, choices)
        else:
            raise ValueError(f"Unsupported variable type: {type(var)}")

        adsg.add_node(var_node)
        nodes[f"x{i}"] = var_node
        start_nodes.add(var_node)

    # Handle decreed variables (conditional dependencies)
    for decreed_var in legacy_ds._cs.conditional_hyperparameters:
        decreed_node = nodes[f"{decreed_var}"]
        if decreed_node in start_nodes:
            start_nodes.remove(decreed_node)
        # Get parent condition(s) from the legacy design space
        parent_conditions = legacy_ds._cs.parent_conditions_of[decreed_var]
        for condition in parent_conditions:
            meta_var = condition.parent.name  # Parent variable
            try:
                meta_values = (
                    condition.values
                )  # Values that activate the decreed variable
            except AttributeError:
                meta_values = [condition.value]

            # Add conditional decreed edges
            for value in meta_values:
                meta_nodes = [node for node in value_nodes[f"{meta_var}"]]
                meta_node_ind = [
                    node.name for node in value_nodes[f"{meta_var}"]
                ].index(str(value)[:])
                value_node = meta_nodes[meta_node_ind]

                nodes[f"x{legacy_ds._cs.index_of[meta_var]}"]
                adsg.add_edge(
                    value_node, decreed_node
                )  # Linking decreed node to meta node

    # Handle value constraints (incompatibilities)
    for value_constraint in legacy_ds._cs.forbidden_clauses:
        clause1 = value_constraint.components[0]
        var1 = clause1.hyperparameter.name
        values1 = getattr(clause1, "values", None) or [getattr(clause1, "value", None)]
        clause2 = value_constraint.components[1]
        var2 = clause2.hyperparameter.name
        values2 = getattr(clause2, "values", None) or [getattr(clause2, "value", None)]

        for value1 in values1:
            for value2 in values2:
                # Retrieve decreed value nodes from value_nodes
                value_nodes1 = [node for node in value_nodes[f"{var1}"]]
                value_node1_ind = [node.name for node in value_nodes[f"{var1}"]].index(
                    str(value1)[:]
                )
                value_node1 = value_nodes1[value_node1_ind]
                value_nodes2 = [node for node in value_nodes[f"{var2}"]]
                value_node2_ind = [node.name for node in value_nodes[f"{var2}"]].index(
                    str(value2)[:]
                )
                value_node2 = value_nodes2[value_node2_ind]
                if value_node1 and value_node2:
                    # Add incompatibility constraint between the two value nodes
                    adsg.add_incompatibility_constraint([value_node1, value_node2])
    adsg = adsg.set_start_nodes(start_nodes)
    return adsg


class AdsgDesignSpaceImpl(BaseDesignSpace):
    """ """

    def __init__(
        self,
        adsg=None,
        design_variables=None,
        seed=None,
    ):
        if adsg is not None:
            self.adsg = adsg
        elif design_variables is not None:
            # to do
            self.ds_leg = ConfigSpaceDesignSpaceImpl(
                design_variables=design_variables, seed=seed
            )
            self.adsg = _legacy_to_adsg(self.ds_leg)
            pass
        else:
            raise ValueError("Either design_variables or adsg should be provided.")

        self.graph_proc = GraphProcessor(graph=self.adsg)

        if not (HAS_ADSG):
            raise ImportError("ADSG is not installed")
        if not (HAS_CONFIG_SPACE):
            raise ImportError("ConfigSpace is not installed")

        design_space = ensure_design_space(design_space=self.adsg)
        self._design_variables = design_space.design_variables
        super().__init__(design_variables=self._design_variables, seed=seed)
        self._cs = design_space._cs
        self._cs_cate = design_space._cs_cate
        self._is_decreed = design_space._is_decreed

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

        self.ds_leg.declare_decreed_var(
            decreed_var=decreed_var, meta_var=meta_var, meta_value=meta_value
        )
        self.adsg = _legacy_to_adsg(self.ds_leg)
        design_space = ensure_design_space(design_space=self.adsg)
        self._design_variables = design_space.design_variables
        self._cs = design_space._cs
        self._cs_cate = design_space._cs_cate
        self._is_decreed = design_space._is_decreed
        self.graph_proc = GraphProcessor(graph=self.adsg)

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

        self.ds_leg.add_value_constraint(
            var1=var1, value1=value1, var2=var2, value2=value2
        )
        self.adsg = _legacy_to_adsg(self.ds_leg)
        design_space = ensure_design_space(design_space=self.adsg)
        self._design_variables = design_space.design_variables
        self._cs = design_space._cs
        self._cs_cate = design_space._cs_cate
        self._is_decreed = design_space._is_decreed
        self.graph_proc = GraphProcessor(graph=self.adsg)

    def _sample_valid_x(
        self,
        n: int,
        seed=None,
        return_render=False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample design vectors"""
        # Get design vectors and get the is_active matrix
        configs0 = []
        configs1 = []
        configs2 = []
        for i in range(n):
            gp_get_i = self.graph_proc.get_graph(
                self.graph_proc.get_random_design_vector(), create=return_render
            )
            configs0.append(gp_get_i[0])
            configs1.append(gp_get_i[1])
            configs2.append(gp_get_i[2])

        if return_render:
            return np.array(configs1), np.array(configs2), np.array(configs0)
        else:
            return np.array(configs1), np.array(configs2)

    def correct_get_acting(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Correct the given matrix of design vectors and return the corrected vectors and the is_acting matrix.
        It is automatically detected whether input is provided in unfolded space or not.

        Parameters
        ----------
        x: np.ndarray [n_obs, dim]
           - Input variables

        Returns
        -------
        x_corrected: np.ndarray [n_obs, dim]
           - Corrected and imputed input variables
        is_acting: np.ndarray [n_obs, dim]
           - Boolean matrix specifying for each variable whether it is acting or non-acting
        """
        return self._correct_x(x)

    def _correct_x(self, x: np.ndarray):
        """
        Fill the activeness matrix (n x nx) and if needed correct design vectors (n x nx) that are partially inactive.
        Imputation of inactive variables is handled automatically.
        """
        x = np.atleast_2d(x)
        is_discrete_mask = self.is_cat_mask
        is_active = np.copy(x)
        for i, xi in enumerate(x):
            x_arch = [
                int(val) if is_discrete_mask[j] else float(val)
                for j, val in enumerate(xi)
            ]
            _, x_imputed, is_active_arch = self.graph_proc.get_graph(
                x_arch, create=False
            )
            x[i, :] = x_imputed
            is_active[i, :] = is_active_arch
        is_active = np.array(is_active, dtype=bool)
        return x, is_active

    def _is_conditionally_acting(self) -> np.ndarray:
        # Decreed variables are the conditionally acting variables
        return np.array(self.graph_proc.dv_is_conditionally_active)
