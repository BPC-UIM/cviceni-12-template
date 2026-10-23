"""Vizualizace SVM: hranice s pasmem a support vektory, Gramova matice, sweep c (PREDVYPLNENO).

Funkce neznaji tridu ``KernelSVM`` (``dataio`` neimportuje ``src``): hranici
kresli z libovolne rozhodovaci funkce ``decision_fn: Tensor (m, 2) -> Tensor (m,)``,
typicky ``model.decision_function``. Vrstevnice ``h(x) = 0`` je hranice,
``h(x) = -1`` a ``h(x) = +1`` jsou okraje pasma.

Vsechny grafy se ukladaji do souboru (backend ``Agg``, bez okna).
"""

from __future__ import annotations

import math
import os
from typing import Callable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import Tensor  # noqa: E402

DecisionFn = Callable[[Tensor], Tensor]
Panel = tuple[str, "DecisionFn | None", "Tensor | None"]

_GRID_RESOLUTION = 200
_COLOR_POS = "#eb6834"         # trida +1 (oranzova, kolecko)
_COLOR_NEG = "#2a78d6"         # trida -1 (modra, ctverecek)
_INK = "#0b0b0b"


def _save(fig: plt.Figure, save_path: str) -> None:
    """Pomocna funkce: vytvori slozku, ulozi obrazek a zavre ho."""
    folder = os.path.dirname(save_path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def _draw_boundary(ax: plt.Axes, decision_fn: DecisionFn | None, x: Tensor, y: Tensor,
                   sv_x: Tensor | None, title: str) -> None:
    """Pomocna funkce: oblasti trid, hranice, pasmo, body a zakrouzkovane support vektory."""
    x_np = np.asarray(x, dtype=np.float64)
    y_np = np.asarray(y, dtype=np.float64)
    pad = 0.5
    x0 = np.linspace(x_np[:, 0].min() - pad, x_np[:, 0].max() + pad, _GRID_RESOLUTION)
    x1 = np.linspace(x_np[:, 1].min() - pad, x_np[:, 1].max() + pad, _GRID_RESOLUTION)
    g0, g1 = np.meshgrid(x0, x1)
    if decision_fn is not None:
        grid = torch.as_tensor(np.column_stack([g0.ravel(), g1.ravel()]), dtype=torch.float64)
        with torch.no_grad():
            h = np.asarray(decision_fn(grid), dtype=np.float64).reshape(g0.shape)
        ax.contourf(g0, g1, np.sign(h), levels=[-2, 0, 2], colors=[_COLOR_NEG, _COLOR_POS],
                    alpha=0.12)
        ax.contour(g0, g1, h, levels=[-1.0, 0.0, 1.0], colors=_INK,
                   linestyles=["--", "-", "--"], linewidths=[1.0, 2.0, 1.0])
    ax.scatter(x_np[y_np > 0, 0], x_np[y_np > 0, 1], s=22, c=_COLOR_POS, marker="o",
               label="+1", edgecolors="white", linewidths=0.5)
    ax.scatter(x_np[y_np < 0, 0], x_np[y_np < 0, 1], s=22, c=_COLOR_NEG, marker="s",
               label="-1", edgecolors="white", linewidths=0.5)
    if sv_x is not None and len(sv_x) > 0:
        sv = np.asarray(sv_x, dtype=np.float64)
        ax.scatter(sv[:, 0], sv[:, 1], s=90, facecolors="none", edgecolors=_INK,
                   linewidths=1.0, label=f"support vektory ({len(sv)})")
    ax.set_xlim(x0[0], x0[-1])
    ax.set_ylim(x1[0], x1[-1])
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x1 (standardizovano)")
    ax.set_ylabel("x2 (standardizovano)")


def plot_decision_boundary(
    decision_fn: DecisionFn, x: Tensor, y: Tensor, save_path: str,
    title: str = "Rozhodovaci hranice SVM", sv_x: Tensor | None = None,
    ) -> None:
    """Hranice ``h = 0`` (plna), pasmo ``h = +-1`` (carkovane) a zakrouzkovane SV.

    Parametry
    ---------
    decision_fn : Callable[[Tensor], Tensor]
        Rozhodovaci funkce ``(m, 2) -> (m,)``, napr. ``model.decision_function``.
    x : Tensor
        Body k vykresleni, tvar ``(n, 2)``.
    y : Tensor
        Tridy ``-1`` / ``+1``, tvar ``(n,)``.
    save_path : str
        Cesta k vystupnimu PNG.
    title : str, optional
        Nadpis grafu.
    sv_x : Tensor | None, optional
        Support vektory k zakrouzkovani, tvar ``(n_sv, 2)``.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    _draw_boundary(ax, decision_fn, x, y, sv_x, title)
    ax.legend(loc="upper right", fontsize=8)
    _save(fig, save_path)


def plot_boundary_panels(
    panels: Sequence[Panel], x: Tensor, y: Tensor, save_path: str,
    suptitle: str = "", ncols: int = 3,
    ) -> None:
    """Nekolik hranic vedle sebe na tychz datech (srovnani modelu, mrizka c x gamma).

    Parametry
    ---------
    panels : Sequence[tuple[str, DecisionFn | None, Tensor | None]]
        Pro kazdy panel: nadpis, rozhodovaci funkce (``None`` = jen body)
        a support vektory (``None`` = nezakrouzkovat).
    x : Tensor
        Body, tvar ``(n, 2)``.
    y : Tensor
        Tridy, tvar ``(n,)``.
    save_path : str
        Cesta k vystupnimu PNG.
    suptitle : str, optional
        Spolecny nadpis.
    ncols : int, optional
        Pocet sloupcu mrizky panelu.
    """
    n = len(panels)
    ncols = max(1, min(ncols, n))
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.8 * nrows), squeeze=False)
    for ax, (title, fn, sv) in zip(axes.ravel(), panels):
        _draw_boundary(ax, fn, x, y, sv, title)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    if suptitle:
        fig.suptitle(suptitle)
    _save(fig, save_path)


def plot_gram_heatmap(k: Tensor, y: Tensor, save_path: str,
                      title: str = "Gramova matice") -> None:
    """Heatmapa matice podobnosti s body serazenymi podle tridy.

    U dobre oddelitelnych trid je videt blokova struktura: body stejne tridy
    jsou si podobne (tmave bloky na diagonale), body ruznych trid ne.

    Parametry
    ---------
    k : Tensor
        Ctvercova matice podobnosti ``K(x, x)``, tvar ``(n, n)``.
    y : Tensor
        Tridy bodu, tvar ``(n,)``.
    save_path : str
        Cesta k vystupnimu PNG.
    title : str, optional
        Nadpis grafu.
    """
    k_np = np.asarray(k, dtype=np.float64)
    y_np = np.asarray(y, dtype=np.float64)
    order = np.argsort(-y_np, kind="stable")
    n_pos = int((y_np > 0).sum())
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(k_np[np.ix_(order, order)], cmap="Blues")
    ax.axhline(n_pos - 0.5, color=_COLOR_POS, lw=1.5)
    ax.axvline(n_pos - 0.5, color=_COLOR_POS, lw=1.5)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(f"body serazene podle tridy (+1: prvnich {n_pos})")
    ax.set_ylabel("body serazene podle tridy")
    fig.colorbar(im, ax=ax, label="K[i, j]")
    _save(fig, save_path)


def plot_dual_history(history: dict[str, list[float]], save_path: str,
                      title: str = "Prubeh treninku v dualu") -> None:
    """Prubeh ``L(lambda)`` a poctu support vektoru po iteracich.

    Parametry
    ---------
    history : dict[str, list[float]]
        Historie z ``KernelSVM.fit`` (klice ``iteration``, ``dual_objective``, ``n_sv``).
    save_path : str
        Cesta k vystupnimu PNG.
    title : str, optional
        Nadpis grafu.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    ax1.plot(history["iteration"], history["dual_objective"], color=_COLOR_NEG, lw=2)
    ax1.set_xlabel("iterace")
    ax1.set_ylabel("L(lambda)")
    ax1.set_title("dualni ucelova funkce (maximalizuje se)", fontsize=10)
    ax1.grid(alpha=0.3)
    ax2.plot(history["iteration"], history["n_sv"], color=_COLOR_NEG, lw=2)
    ax2.set_xlabel("iterace")
    ax2.set_ylabel("pocet support vektoru")
    ax2.set_title("support vektory (lambda > prah)", fontsize=10)
    ax2.grid(alpha=0.3)
    fig.suptitle(title)
    _save(fig, save_path)


def plot_c_sweep(rows: Sequence[dict], save_path: str,
                 title: str = "Vliv regularizace c") -> None:
    """Sirka pasma, pocet support vektoru a presnost v zavislosti na ``c`` (log. osa).

    Parametry
    ---------
    rows : Sequence[dict]
        Vystup ``sweep_c`` (klice ``c``, ``margin``, ``n_sv``, ``acc_train``, ``acc_test``).
    save_path : str
        Cesta k vystupnimu PNG.
    title : str, optional
        Nadpis grafu.
    """
    c = [r["c"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    axes[0].plot(c, [r["margin"] for r in rows], "o-", color=_COLOR_NEG, lw=2)
    axes[0].set_ylabel("sirka pasma 2/||w||")
    axes[1].plot(c, [r["n_sv"] for r in rows], "o-", color=_COLOR_NEG, lw=2)
    axes[1].set_ylabel("pocet support vektoru")
    axes[2].plot(c, [r["acc_train"] for r in rows], "o-", color=_COLOR_NEG, lw=2,
                 label="trenink")
    axes[2].plot(c, [r["acc_test"] for r in rows], "s--", color=_COLOR_POS, lw=2,
                 label="test")
    axes[2].set_ylabel("presnost")
    axes[2].legend()
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("c")
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    _save(fig, save_path)
