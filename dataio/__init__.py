"""Verejne API balicku ``dataio`` pro cviceni 12.

Datove sady (2D hracky a breast cancer), rozdeleni se standardizaci,
typovana konfigurace a vykreslovani.

Verejne API
-----------
- ``load_blobs`` / ``load_circles`` / ``load_moons`` -- 2D hracky ze ``sklearn``
- ``load_breast_cancer`` -- realna uloha, 30 priznaku (``+1`` = maligni)
- ``load_iris_binary`` -- volitelne: dve tridy kosatcu, 4 priznaky
- ``train_test_split_std`` -- stratifikovane rozdeleni + standardizace z treninku
- ``load_config`` / ``validate_config`` -- typovana konfigurace nad ``config.yaml``
- ``PathsConfig`` / ``DataConfig`` / ``KernelConfig`` / ``KernelsConfig`` /
  ``TrainingConfig`` / ``SweepConfig`` / ``DiagnosticsConfig`` / ``ExperimentConfig``
  -- dataclassy konfigurace (vse nastavitelne je v ``config.yaml``)
- ``KERNEL_NAMES`` -- povolena jmena jader
- ``plot_decision_boundary`` -- hranice, pasmo a support vektory
- ``plot_boundary_panels`` -- vice hranic vedle sebe (srovnani, mrizka c x gamma)
- ``plot_gram_heatmap`` -- heatmapa Gramovy matice serazena podle trid
- ``plot_dual_history`` -- prubeh ``L(lambda)`` a poctu support vektoru
- ``plot_c_sweep`` -- sirka pasma, pocet SV a presnost v zavislosti na ``c``

Cely balicek ``dataio/`` je v tomto cviceni **predvyplneny** -- zadny
studentsky ukol, zadny ``NotImplementedError``.

**Tento soubor neupravujte.**
"""

from __future__ import annotations

from dataio.config_manager import (
    KERNEL_NAMES,
    DataConfig,
    DiagnosticsConfig,
    ExperimentConfig,
    KernelConfig,
    KernelsConfig,
    PathsConfig,
    SweepConfig,
    TrainingConfig,
    load_config,
    validate_config,
)
from dataio.loaders import (
    load_blobs,
    load_breast_cancer,
    load_circles,
    load_iris_binary,
    load_moons,
    train_test_split_std,
)
from dataio.plotting import (
    plot_boundary_panels,
    plot_c_sweep,
    plot_decision_boundary,
    plot_dual_history,
    plot_gram_heatmap,
)

__all__ = [
    "load_blobs",
    "load_circles",
    "load_moons",
    "load_breast_cancer",
    "load_iris_binary",
    "train_test_split_std",
    "load_config",
    "validate_config",
    "KERNEL_NAMES",
    "PathsConfig",
    "DataConfig",
    "KernelConfig",
    "KernelsConfig",
    "TrainingConfig",
    "SweepConfig",
    "DiagnosticsConfig",
    "ExperimentConfig",
    "plot_decision_boundary",
    "plot_boundary_panels",
    "plot_gram_heatmap",
    "plot_dual_history",
    "plot_c_sweep",
]
