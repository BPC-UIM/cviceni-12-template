"""Spolecne kroky pipeline: sestaveni modelu z konfigurace, trenink a kontrola (PREDVYPLNENO).

Jde o orchestraci pipeline ``cviceni_12.py``, kterou student neupravuje.
Lezi mimo ``src/`` (studentske ukoly) a ``dataio/`` (data, konfigurace,
grafy); smer zavislosti je ``utils`` -> ``src``, ``dataio`` (nikdy naopak).

Prah kontroly konvergence se cte ze sekce ``diagnostics`` v ``config.yaml``.
"""

from __future__ import annotations

import math
import os
import time
from typing import Callable

import torch
from sklearn.linear_model import Perceptron
from torch import Tensor

from dataio import DiagnosticsConfig, KernelConfig, TrainingConfig, train_test_split_std
from src import Kernel, KernelSVM, get_logger, make_kernel
from src.svm import LOG_FILE_NAME
from utils.containers import DatasetBundle, SweepRow, TrainedModel


class UceniDivergovalo(RuntimeError):
    """Dualni ucelova funkce behem treninku neni konecna nebo klesla pod pocatecni hodnotu."""


def rozdel(x: Tensor, y: Tensor, name: str, test_size: float, seed: int) -> DatasetBundle:
    """Rozdeli data (stratifikovane) a standardizuje je statistikami z treninku.

    Parametry
    ---------
    x : Tensor
        Surove priznaky, tvar ``(n, d)``.
    y : Tensor
        Tridy ``-1`` / ``+1``, tvar ``(n,)``.
    name : str
        Jmeno ulohy (do vypisu a nazvu grafu).
    test_size : float
        Podil testovacich bodu (``data.test_size``).
    seed : int
        Seed rozdeleni.

    Navratova hodnota
    -----------------
    DatasetBundle
        Rozdelena data.
    """
    x_train, y_train, x_test, y_test = train_test_split_std(x, y, test_size=test_size, seed=seed)
    return DatasetBundle(x_train, y_train, x_test, y_test, name)


def priprav_logovani(logs_dir: str) -> str:
    """Nakonfiguruje logger treninku (konzole + soubor) a vrati cestu k souboru logu.

    Parametry
    ---------
    logs_dir : str
        Slozka pro log (``paths.logs_dir``).

    Navratova hodnota
    -----------------
    str
        Cesta k souboru ``svm.log``.
    """
    get_logger(logs_dir)
    return os.path.join(logs_dir, LOG_FILE_NAME)


def jadro(kernel_cfg: KernelConfig) -> Kernel:
    """Sestavi jadro podle konfigurace faze tovarni funkci ``make_kernel``.

    Parametry
    ---------
    kernel_cfg : KernelConfig
        Sekce ``kernels.<faze>`` z ``config.yaml``.

    Navratova hodnota
    -----------------
    Kernel
        Instance zvoleneho jadra.
    """
    return make_kernel(kernel_cfg.name, r=kernel_cfg.r, gamma=kernel_cfg.gamma)


def tovarna_modelu(kernel: Kernel, training: TrainingConfig) -> Callable[[float], KernelSVM]:
    """Vrati funkci ``c -> KernelSVM`` (nenatrenovany model s danym jadrem).

    Parametry
    ---------
    kernel : Kernel
        Hotove jadro (dependency injection -- predava se vsem modelum).
    training : TrainingConfig
        Hyperparametry treninku; ``c`` se bere z argumentu vracene funkce.

    Navratova hodnota
    -----------------
    Callable[[float], KernelSVM]
        Tovarna modelu pro ``sweep_c`` i ``train_and_eval``.
    """
    def make_model(c: float) -> KernelSVM:
        return KernelSVM(kernel, c=c, lr=training.lr, n_iters=training.n_iters,
                         sv_threshold=training.sv_threshold)
    return make_model


def kontrola_uceni(history: dict[str, list[float]], diagnostics: DiagnosticsConfig) -> None:
    """Zkontroluje prubeh maximalizace dualu: divergence a nedokoncena konvergence.

    * ``L(lambda)`` neni konecne, nebo konecna hodnota je NIZSI nez po 1. iteraci
      (vystup ma ``L`` zvysovat) -> ``UceniDivergovalo`` (krok ``lr`` je prilis velky).
    * ``L`` za poslednich 10 % iteraci vzrostlo relativne o vic nez
      ``diagnostics.convergence_rtol`` -> ``[POZOR]`` (trenink jeste nedobehl).

    Parametry
    ---------
    history : dict[str, list[float]]
        Historie z ``KernelSVM.fit``.
    diagnostics : DiagnosticsConfig
        Prahy kontrol z ``config.yaml``.

    Vyjimky
    -------
    UceniDivergovalo
        Pri divergenci treninku.
    """
    values = history["dual_objective"]
    if not all(math.isfinite(v) for v in values) or values[-1] < values[0]:
        raise UceniDivergovalo(
            f"L(lambda) neroste (po 1. iteraci {values[0]:.4g}, na konci {values[-1]:.4g}) "
            "-- krok je prilis velky; zmensete training.lr nebo nastavte lr: auto"
        )
    n = len(values)
    start = values[max(0, n - 1 - max(1, n // 10))]
    narust = (values[-1] - start) / max(abs(values[-1]), 1e-12)
    if narust > diagnostics.convergence_rtol:
        print(f"  [POZOR] L(lambda) za poslednich 10 % iteraci vzrostlo o {100 * narust:.1f} % "
              "-- trenink jeste nedokonvergoval; zvyste training.n_iters "
              f"(prah diagnostics.convergence_rtol = {diagnostics.convergence_rtol}).")


def train_and_eval(make_model: Callable[[float], KernelSVM], bundle: DatasetBundle,
                   c: float, diagnostics: DiagnosticsConfig,
                   vypis_cas: bool = True) -> TrainedModel:
    """Vytvori model, natrenuje ho, zkontroluje prubeh a spocita metriky.

    Parametry
    ---------
    make_model : Callable[[float], KernelSVM]
        Tovarna ``c -> KernelSVM`` (napr. z ``tovarna_modelu``).
    bundle : DatasetBundle
        Rozdelena data.
    c : float
        Regularizace modelu (``training.c``).
    diagnostics : DiagnosticsConfig
        Prahy kontrol z ``config.yaml``.
    vypis_cas : bool, optional
        Vypsat dobu treninku (``False`` pri mnoha treninku za sebou, napr. v mrizce).

    Navratova hodnota
    -----------------
    TrainedModel
        Model, pocet support vektoru a presnost na treninku i testu.

    Vyjimky
    -------
    UceniDivergovalo
        Pri divergenci treninku.
    """
    start = time.perf_counter()
    model = make_model(c).fit(bundle.x_train, bundle.y_train)
    if vypis_cas:
        print(f"  doba treninku: {time.perf_counter() - start:.1f} s "
              f"(krok vystupu {model.step_:.2e})")
    kontrola_uceni(model.history_, diagnostics)
    return TrainedModel(model, len(model.support_), model.score(bundle.x_train, bundle.y_train),
                        model.score(bundle.x_test, bundle.y_test))


def nauc_perceptron(bundle: DatasetBundle, seed: int) -> tuple[Callable[[Tensor], Tensor], float]:
    """Referencni perceptron (``sklearn``) pro srovnani se SVM.

    Perceptron najde *nejakou* oddelujici primku (bez pojmu pasma); slouzi jen
    jako kontrast, neni to studentsky ukol.

    Parametry
    ---------
    bundle : DatasetBundle
        Rozdelena data.
    seed : int
        Seed michani v perceptronu.

    Navratova hodnota
    -----------------
    tuple[Callable[[Tensor], Tensor], float]
        Rozhodovaci funkce ``(m, d) -> (m,)`` a presnost na testu.
    """
    perceptron = Perceptron(random_state=seed).fit(bundle.x_train.numpy(), bundle.y_train.numpy())

    def decision_fn(x: Tensor) -> Tensor:
        return torch.as_tensor(perceptron.decision_function(x.numpy()), dtype=torch.float64)

    return decision_fn, float(perceptron.score(bundle.x_test.numpy(), bundle.y_test.numpy()))


def radky_sweepu(rows: list[dict]) -> list[SweepRow]:
    """Prevede vystup ``sweep_c`` (seznam slovniku) na seznam ``SweepRow``.

    Parametry
    ---------
    rows : list[dict]
        Slovniky s klici ``c``, ``margin``, ``n_sv``, ``acc_train``, ``acc_test``.

    Navratova hodnota
    -----------------
    list[SweepRow]
        Tytez hodnoty jako dataclassy.

    Vyjimky
    -------
    ``KeyError``:
        Pokud nektery slovnik nema predepsany klic.
    """
    return [SweepRow(float(r["c"]), float(r["margin"]), int(r["n_sv"]),
                     float(r["acc_train"]), float(r["acc_test"])) for r in rows]
