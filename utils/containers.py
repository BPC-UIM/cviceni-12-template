"""Prepravky na vysledky fazi pipeline (PREDVYPLNENO).

Jde o orchestraci pipeline ``cviceni_12.py``, kterou student neupravuje.
Prepravky jen drzi data, ktera si faze predavaji (rozdelena data,
natrenovany model a jeho metriky, radek sweepu, metriky klasifikace). Lezi mimo ``src/``
a ``dataio/``; smer zavislosti je ``utils`` -> ``src``, ``dataio`` (nikdy naopak).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from torch import Tensor

from src import KernelSVM


@dataclass
class DatasetBundle:
    """Rozdelena a standardizovana data jedne ulohy."""

    x_train: Tensor                # (n_train, d), standardizovano statistikami z treninku
    y_train: Tensor                # (n_train,), tridy -1 / +1
    x_test: Tensor                 # (n_test, d)
    y_test: Tensor                 # (n_test,)
    name: str                      # jmeno ulohy do vypisu a nazvu grafu


@dataclass
class TrainedModel:
    """Natrenovane SVM a jeho zakladni metriky."""

    model: KernelSVM
    n_sv: int                      # pocet support vektoru
    acc_tr: float                  # presnost na trenovacich datech
    acc_te: float                  # presnost na testovacich datech


@dataclass
class SweepRow:
    """Jeden radek experimentu s regularizaci (vystup ``sweep_c`` jako dataclass)."""

    c: float
    margin: float                  # sirka pasma 2 / ||w||
    n_sv: int
    acc_tr: float
    acc_te: float


@dataclass
class KlasifikacniMetriky:
    """Kontingencni tabulka a z ni odvozene metriky (vystup ``vyhodnot_klasifikaci``)."""

    cm: np.ndarray                 # (2, 2), [[TN, FP], [FN, TP]], pozitivni trida = +1
    accuracy: float
    precision: float
    sensitivity: float             # recall = Se = TP / (TP + FN)
    specificity: float             # Sp = TN / (TN + FP)
    f1: float
