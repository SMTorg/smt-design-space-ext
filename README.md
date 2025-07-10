
# SMT Design Space Extension 🌳

**Enhancing SMT with Hierarchical & Conditional Design Spaces**

This package is an extension to the SMT (Surrogate Modeling Toolbox), offering a powerful framework to define and manage **hierarchical**, **mixed-type**, and **conditionally active** variables in design spaces.

---

## 🔍 What It Does

- **Hierarchical variables**: Support for nested conditional variables (e.g., a rotor configuration branch that only activates when `use_rotor = yes`).
- **Mixed types**: Handles continuous, integer, categorical, meta and indicator variables uniformly.
- **Conditional activation**: Meta-variables powerfully control lower-level variable activation based on context.
- **Graph-based design space representation**: Clean and intuitive implementation of complex, branching designs.
- **Extensible subtress**: Easily add new types or layers of conditional logic.

---

## 📖 Why It Matters

In many engineering and simulation contexts, the design parameters form a structured, dependent hierarchy—optimizing them without a formal framework can lead to errors or inefficient models. This extension provides:

- A **formal representation** of hierarchical dependencies.
- Seamless integration with surrogate modeling tools and optimization routines.
- A foundation for improved experimental design and architectural exploration.

---

## 🔗 Reference

This implementation is based on:

> **Hierarchical Modeling and Architecture Optimization: Review and Unified Framework**  
> P. Saves, E. Hallé‑Hannan, J. Bussemaker, Y. Diouane, N. Bartoli (June 2025). arXiv:2506.22621 ([arXiv](https://arxiv.org/abs/2506.22621))

The paper provides a comprehensive survey and introduces a unified graph-based model that serves as the theoretical foundation for this extension.

---

## 📦 Installation

```bash
pip install smt-design-space-ext
```

---

## 🚀 Quick Start Example

```python
from smt_design_space import DesignSpace

ds = DesignSpace()
# Meta-variable to enable feature branching
ds.add_categorical("enable_feature", ["yes", "no"])
# Feature type only active if enabled
ds.add_meta("feature_type",
            parent="enable_feature",
            active_if="yes",
            values=["typeA", "typeB"])
# Parameter only active in typeA branch
ds.add_numeric("featureA_param",
               parent="feature_type",
               active_if="typeA",
               bounds=(0.0, 1.0))
```

This dynamically builds a tree-like structure of variables, enabling clear and constrained space exploration.

---

## 🧩 Integration

- Integrates readily with SMT’s Kriging and PyKriging modules.
- Compatible with Bayesian, evolutionary, or gradient‑based optimizers.
- Prepare space definitions for use with graph neural networks or surrogate modeling pipelines.

---

## 📚 Additional Resources

- The **SMT repository** contains tutorials and example notebooks to demonstrate usage.
- Issue tracker and usage discussions can be found under `SMTorg/smt-design-space-ext`.
- Author metadata and license summary confirm this is BSD‑licensed.

---

## ✅ Citation

If you use this package in research, please cite:

```
P. Saves, E. Hallé‑Hannan, J. Bussemaker, Y. Diouane, N. Bartoli,
“Hierarchical Modeling and Architecture Optimization: Review and Unified Framework,”
arXiv:2506.22621, June 2025.
```

---

## 📜 License

Distributed under the **BSD-3-Clause** license. See [LICENSE](LICENSE) for details.
