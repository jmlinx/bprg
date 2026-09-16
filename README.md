# BPRG: Bootstrap Percolation in Random Graphs

[![PyPI](https://img.shields.io/pypi/v/bprg?logo=pypi&logoColor=white)](https://pypi.org/project/bprg/)
[![Python](https://img.shields.io/pypi/pyversions/bprg?logo=python&logoColor=white)](https://pypi.org/project/bprg/)
[![License](https://img.shields.io/github/license/jmlinx/bprg)](https://github.com/jmlinx/bprg/blob/main/LICENSE)
[![Documentation](https://img.shields.io/badge/Documentation-Zensical-007EC6?logo=materialformkdocs&logoColor=white)](https://jmlinx.github.io/bprg/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Demo-FF4B4B?logo=streamlit&logoColor=white)](https://bprgur.streamlit.app/)

[bootstrap_percolation.mp4](https://github.com/user-attachments/assets/99ba0cee-6c88-4e8d-b9ad-d01964981a2a)

`BPRG` is a high performance Python library for analyzing **bootstrap percolation in random graphs**. It implements the theoretical framework in [Detering and Lin (2026): Bootstrap percolation in random graphs of unbounded rank](https://doi.org/10.1017/apr.2026.10072), providing comprehensive tools to study contagion dynamics in random graphs.



## Installation

Install `bprg` via [uv](https://docs.astral.sh/uv/) (recommended):

```bash
uv add bprg
```

or via [pip](https://pypi.org/project/pip/):

```bash
pip install bprg
```


## Interactive Demo

Explore the interactive analysis in the [BPRG Streamlit app](https://bprgur.streamlit.app/).

The demo provides an intuitive interface to simulate random graphs, visualize the bootstrap percolation process, and estimate the final fraction of infected vertices using both the asymptotic fixed point solution and Monte Carlo simulation.


## Overview

### Bootstrap percolation in random graphs

**Bootstrap percolation** is a contagion process on graphs where vertices have *thresholds*. The process starts with a set of initially infected vertices. An uninfected vertex with threshold $k$ becomes *infected* as soon as it has *at least $k$ infected neighbors*. This simple rule gives rise to complex infection dynamics.

For the **Random Graphs** setting, `BPRG` considers those of *unbounded rank*, which admits a *general kernel function* $κ$ that determines connection probabilities by the *types* of vertices.

### Final fraction of infected vertices
The *final fraction of infected vertices* is a key quantity of interest in bootstrap percolation. It represents the proportion of the graph that becomes infected in the long run, after the infection process has stabilized. `BPRG` provides tools to compute this quantity via both *theoretical asymptotic solution* and *Monte Carlo simulation*.

#### Theoretical asymptotic solution

One of the main theoretical results of [Detering and Lin (2026)](https://doi.org/10.1017/apr.2026.10072) is that the asymptotic final fraction of infected vertices in large random graphs is determined by the *least fixed point* of a non-linear operator on the space of functions. `BPRG` provides **neural network-based solvers** to promptly compute this fixed point.

#### Monte Carlo simulation

`BPRG` also provides a **Monte Carlo simulation framework** to efficiently generate random graphs, simulate bootstrap percolation in the graph, and estimate the final fraction of infected vertices. This allows users to validate theoretical predictions and explore infection dynamics in finite-size graphs.


## Key Features

| Feature                              | Description                                                          |
| :----------------------------------- | :------------------------------------------------------------------- |
| **JAX-based Implementation**         | High-performance computations leveraging JAX                         |
| **Random Graph of Unbounded Rank**   | Generate random graphs from general kernel functions $κ$             |
| **Bootstrap Percolation Simulation** | Simulate infection dynamics with efficient                           |
| **Neutral Fixed Point Solvers**      | Neural network-based operator solvers for asymptotic final infection |
| **Monte Carlo Estimation**           | Estimate final infection through Monte Carlo methods                 |


## References

```
@article{detering2026bootstrap,
  title={Bootstrap Percolation in Random Graphs of Unbounded Rank},
  author={Detering, Nils and Lin, Jimin},
  journal={Advances in Applied Probability},
  year={2026},
  doi={doi.org/10.1017/apr.2026.10072}
}
```
