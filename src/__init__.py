"""Verejne API balicku ``src`` pro cviceni 12 (SVM a kernelovy trik v PyTorch).

Obsahuje:

* ``Kernel`` -- spolecne rozhrani jader: ``gram(x, z)`` vraci matici podobnosti
  (predvyplneno).
* ``LinearKernel`` -- linearni jadro ``a^T b`` (VZOR, predvyplneno).
* ``PolynomialKernel`` -- ``(a^T b)^r`` a explicitni ``feature_map`` pro ``r = 2`` (UKOL).
* ``RBFKernel`` -- ``exp(-gamma ||a - b||^2)`` (UKOL).
* ``make_kernel`` -- tovarni funkce jader podle jmena z ``config.yaml`` (UKOL).
* ``verify_kernel_trick`` -- overeni ``K = phi . phi`` pro polynom stupne 2 (UKOL).
* ``KernelSVM`` -- SVM v dualu: smycka ``fit``, ``predict``, ``score`` a logovani
  (predvyplneno); ``dual_objective``, ``_ascent_step``, ``support_vector_indices``,
  ``decision_function``, ``save`` a ``load`` (UKOL).
* ``sweep_c`` -- vliv regularizace ``c`` na pasmo, support vektory a presnost (UKOL).
* ``get_logger`` -- konfigurace loggeru treninku (predvyplneno).

**Tento soubor neupravujte.**
"""

from __future__ import annotations

from src.kernels import (
    Kernel,
    LinearKernel,
    PolynomialKernel,
    RBFKernel,
    make_kernel,
    verify_kernel_trick,
)
from src.svm import KernelSVM, get_logger, sweep_c

__all__ = [
    "Kernel",
    "LinearKernel",
    "PolynomialKernel",
    "RBFKernel",
    "make_kernel",
    "verify_kernel_trick",
    "KernelSVM",
    "sweep_c",
    "get_logger",
]
