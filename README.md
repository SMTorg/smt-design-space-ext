
# SMT Design Space Extension 🌳

**Enhancing SMT with Hierarchical & Conditional Design Spaces**

This package is an extension to the SMT (Surrogate Modeling Toolbox), offering a powerful framework to define and manage **hierarchical**, **mixed-type**, and **conditionally active** variables in design spaces.

---

## 🔍 What It Does

- **Hierarchical variables**: Support for nested conditional variables (e.g., a rotor configuration branch that only activates when `use_rotor = yes`).
- **Mixed types**: Handles continuous, integer or categorical variables uniformly.
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
https://pypi.org/project/smt-design-space-ext/

Requirements. See [Requirements](requirements.txt) for details.

---

## 🚀 Quick Start Example

```python
from smt_design_space_ext import (
    HAS_CONFIG_SPACE,
    HAS_ADSG,
    AdsgDesignSpaceImpl,
    ConfigSpaceDesignSpaceImpl,
    BaseDesignSpace,
    CategoricalVariable,
    FloatVariable,
    IntegerVariable,
    OrdinalVariable,
)

 ds = ConfigSpaceDesignSpaceImpl(
            [
                CategoricalVariable(["A", "B", "C"]),  # x0
                CategoricalVariable(["E", "F"]),  # x1
                IntegerVariable(0, 1),  # x2
                FloatVariable(0.1, 1),  # x3
            ],
            seed=42,
        )
        ds.declare_decreed_var(
            decreed_var=3, meta_var=0, meta_value="A"
        )  # Activate x3 if x0 == A


```

This dynamically builds a tree-like structure of variables, enabling clear and constrained space exploration.

---

## 🧩 Integration

- Integrates readily with SMT’s Kriging modules.
- Compatible with Bayesian or gradient‑based optimizers.
- Prepare space definitions for use with surrogate modeling pipelines.

---

## 📚 Additional Resources

- The **SMT repository** contains tutorials and example notebooks to demonstrate usage.
- Issue tracker and usage discussions can be found under `SMTorg/smt-design-space-ext`.
- Author metadata and license summary confirm this is BSD‑licensed.

---

## ✅ Citation

## ✅ Citation

If you use this package in research, please cite our published paper:

> P. Saves, E. Hallé-Hannan, J. Bussemaker, Y. Diouane, and N. Bartoli, "Modeling Hierarchical Spaces: A Review and Unified Framework for Surrogate-Based Architecture Design," *Structural and Multidisciplinary Optimization*, vol. 69, no. 3, p. 65, Feb. 2026. DOI: [10.1007/s00158-026-04249-2](https://doi.org/10.1007/s00158-026-04249-2).

```bibtex
@article{saves2026modeling,
  title={Modeling hierarchical spaces: a review and unified framework for surrogate-based architecture design},
  author={Saves, Paul and Hall{\'e}-Hannan, Edward and Bussemaker, Jasper and Diouane, Youssef and Bartoli, Nathalie},
  journal={Structural and Multidisciplinary Optimization},
  volume={69},
  number={3},
  pages={65},
  year={2026},
  publisher={Springer Nature},
  doi={10.1007/s00158-026-04249-2},
  url={[https://doi.org/10.1007/s00158-026-04249-2](https://doi.org/10.1007/s00158-026-04249-2)}
}
---

## 📜 License

Distributed under the **BSD-3-Clause** license. See [LICENSE](LICENSE) for details.
