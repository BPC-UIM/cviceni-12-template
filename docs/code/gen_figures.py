# -*- coding: utf-8 -*-
"""Autorsky skript: generuje obrazky do docs/img/ pro README (NENI studentsky ukol).

Spousteni z korene repozitare:  python docs/code/gen_figures.py

Skript je zamerne nezavisly na src/ (studentske stuby): jadra pocita primo
v NumPy a SVM trenuje pres ``sklearn.svm.SVC`` (standardni formulace s biasem b,
na obrazcich k nerozeznani od dualu s foldnutym biasem ze src/svm.py).
Obrazky se verzuji (docs/img/ neni v .gitignore).
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.datasets import make_blobs, make_circles, make_moons  # noqa: E402
from sklearn.linear_model import Perceptron  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "img")
POS, NEG, INK, MUTED = "#eb6834", "#2a78d6", "#0b0b0b", "#898781"
SEED = 42

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": "#52514e",
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 11,
})


def _save(fig: plt.Figure, name: str) -> None:
    os.makedirs(OUT, exist_ok=True)
    if fig.get_layout_engine() is None:
        fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=130)
    plt.close(fig)
    print("ulozeno:", os.path.join("docs", "img", name))


def _std(x: np.ndarray) -> np.ndarray:
    return StandardScaler().fit_transform(x)


def _labels(y: np.ndarray) -> np.ndarray:
    return np.where(y == 1, 1, -1)


def _points(ax: plt.Axes, x: np.ndarray, y: np.ndarray, s: int = 22) -> None:
    ax.scatter(x[y > 0, 0], x[y > 0, 1], s=s, c=POS, marker="o", edgecolors="white",
               linewidths=0.5, label="třída +1")
    ax.scatter(x[y < 0, 0], x[y < 0, 1], s=s, c=NEG, marker="s", edgecolors="white",
               linewidths=0.5, label="třída −1")


def _grid(x: np.ndarray, pad: float = 0.5, n: int = 250):
    x0 = np.linspace(x[:, 0].min() - pad, x[:, 0].max() + pad, n)
    x1 = np.linspace(x[:, 1].min() - pad, x[:, 1].max() + pad, n)
    return np.meshgrid(x0, x1)


def _boundary(ax: plt.Axes, decision, x: np.ndarray, y: np.ndarray, title: str,
              sv: np.ndarray | None = None, margin: bool = True) -> None:
    g0, g1 = _grid(x)
    h = decision(np.column_stack([g0.ravel(), g1.ravel()])).reshape(g0.shape)
    ax.contourf(g0, g1, np.sign(h), levels=[-2, 0, 2], colors=[NEG, POS], alpha=0.12)
    levels = [-1, 0, 1] if margin else [0]
    styles = ["--", "-", "--"] if margin else ["-"]
    ax.contour(g0, g1, h, levels=levels, colors=INK, linestyles=styles,
               linewidths=[1.0, 2.0, 1.0] if margin else [2.0])
    _points(ax, x, y)
    if sv is not None:
        ax.scatter(sv[:, 0], sv[:, 1], s=95, facecolors="none", edgecolors=INK, linewidths=1.0,
                   label=f"support vektory ({len(sv)})")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


# --------------------------------------------------------------------------- #
def fig_pasmo() -> None:
    """Perceptron najde nejakou primku, SVM tu s nejsirsim pasmem."""
    x, y = make_blobs(n_samples=60, centers=2, cluster_std=0.9, random_state=6)
    x, y = _std(x), _labels(y)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    g0, g1 = _grid(x)
    grid = np.column_stack([g0.ravel(), g1.ravel()])
    for seed in range(6):
        p = Perceptron(random_state=seed, shuffle=True, max_iter=1000).fit(x, y)
        h = p.decision_function(grid).reshape(g0.shape)
        axes[0].contour(g0, g1, h, levels=[0], colors=MUTED, linewidths=1.2)
    _points(axes[0], x, y)
    axes[0].set_title("Perceptron: některá oddělující přímka\n(6 běhů = 6 různých přímek)")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    svm = SVC(kernel="linear", C=1e3).fit(x, y)
    _boundary(axes[1], svm.decision_function, x, y,
              "SVM: přímka s nejširším pásmem\n(pásmo určují jen support vektory)",
              sv=svm.support_vectors_)
    w = svm.coef_[0]
    width = 2.0 / np.linalg.norm(w)
    axes[1].text(0.02, 0.03, f"šířka pásma 2/‖w‖ = {width:.2f}", transform=axes[1].transAxes,
                 color=INK)
    axes[1].legend(loc="upper right", fontsize=8)
    _save(fig, "pasmo_perceptron_vs_svm.png")


def fig_soft_margin() -> None:
    """Prekryvajici se tridy: male C (mekke siroke pasmo) vs. velke C (tvrde uzke)."""
    x, y = make_blobs(n_samples=120, centers=2, cluster_std=2.2, random_state=3)
    x, y = _std(x), _labels(y)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    for ax, c in zip(axes, (0.01, 1.0, 100.0)):
        svm = SVC(kernel="linear", C=c).fit(x, y)
        width = 2.0 / np.linalg.norm(svm.coef_[0])
        _boundary(ax, svm.decision_function, x, y,
                  f"C = {c:g}: pásmo {width:.2f}, SV {len(svm.support_)}",
                  sv=svm.support_vectors_)
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Soft-margin: malé C toleruje body v pásmu (široké měkké pásmo), "
                 "velké C je trestá (úzké tvrdé pásmo)")
    _save(fig, "soft_margin_c.png")


def fig_transformace() -> None:
    """Trojice: puvodni 2D, skutecny 3D lift polynomu st. 2, kernel PCA pro RBF."""
    x, y = make_circles(n_samples=300, noise=0.05, factor=0.3, random_state=SEED)
    x, y = _std(x), _labels(y)
    fig = plt.figure(figsize=(15, 4.8), layout="constrained")

    ax1 = fig.add_subplot(1, 3, 1)
    _points(ax1, x, y)
    ax1.set_title("1) Původní 2D: přímkou neoddělitelné")
    ax1.set_xlabel("x₁")
    ax1.set_ylabel("x₂")
    ax1.legend(loc="upper right", fontsize=8)

    phi = np.column_stack([x[:, 0] ** 2, np.sqrt(2) * x[:, 0] * x[:, 1], x[:, 1] ** 2])
    lin = SVC(kernel="linear", C=10.0).fit(phi, y)
    w, b = lin.coef_[0], lin.intercept_[0]
    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    ax2.scatter(phi[y > 0, 0], phi[y > 0, 1], phi[y > 0, 2], c=POS, s=12, marker="o")
    ax2.scatter(phi[y < 0, 0], phi[y < 0, 1], phi[y < 0, 2], c=NEG, s=12, marker="s")
    z1, z2 = np.meshgrid(np.linspace(0, phi[:, 0].max(), 10),
                         np.linspace(phi[:, 1].min(), phi[:, 1].max(), 10))
    z3 = -(b + w[0] * z1 + w[1] * z2) / w[2]
    z3 = np.where((z3 >= 0) & (z3 <= phi[:, 2].max()), z3, np.nan)
    ax2.plot_surface(z1, z2, z3, color=MUTED, alpha=0.35)
    ax2.set_xlabel("x₁²")
    ax2.set_ylabel("√2·x₁x₂")
    ax2.set_zlabel("x₂²")
    ax2.view_init(elev=18, azim=-60)
    ax2.set_title("2) Polynom st. 2: skutečný prostor φ (3D)\noddělí ROVINA")

    gamma = 2.0
    d2 =((x[:, None, :] - x[None, :, :]) ** 2).sum(-1)
    k = np.exp(-gamma * d2)
    n = len(x)
    centering = np.eye(n) - np.ones((n, n)) / n
    vals, vecs = np.linalg.eigh(centering @ k @ centering)
    idx = np.argsort(vals)[::-1][:2]
    z = vecs[:, idx] * np.sqrt(vals[idx])
    ax3 = fig.add_subplot(1, 3, 3)
    _points(ax3, z, y)
    ax3.set_title(f"3) RBF (γ = {gamma:g}): kernel PCA, 2 hlavní komponenty\n"
                  "jen STÍN ∞-rozměrného prostoru, přesto oddělitelné")
    ax3.set_xlabel("1. komponenta")
    ax3.set_ylabel("2. komponenta")
    _save(fig, "transformace_trojice.png")


def fig_srovnani() -> None:
    """Pulmesice: perceptron vs. linearni SVM vs. RBF SVM."""
    x, y = make_moons(n_samples=200, noise=0.15, random_state=SEED)
    x, y = _std(x), _labels(y)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    p = Perceptron(random_state=SEED).fit(x, y)
    _boundary(axes[0], p.decision_function, x, y,
              f"Perceptron: přesnost {p.score(x, y):.2f}", margin=False)
    lin = SVC(kernel="linear", C=1.0).fit(x, y)
    _boundary(axes[1], lin.decision_function, x, y,
              f"Lineární SVM: přesnost {lin.score(x, y):.2f}", sv=lin.support_vectors_)
    rbf = SVC(kernel="rbf", gamma=1.0, C=1.0).fit(x, y)
    _boundary(axes[2], rbf.decision_function, x, y,
              f"RBF SVM (γ = 1, C = 1): přesnost {rbf.score(x, y):.2f}",
              sv=rbf.support_vectors_)
    axes[0].legend(loc="upper right", fontsize=8)
    _save(fig, "srovnani_hranic.png")


def fig_vliv_c_gamma() -> None:
    """Mrizka C x gamma pro RBF na pulmesicich."""
    x, y = make_moons(n_samples=200, noise=0.25, random_state=SEED)
    x, y = _std(x), _labels(y)
    gammas, cs = (0.1, 1.0, 30.0), (0.1, 1.0, 100.0)
    fig, axes = plt.subplots(3, 3, figsize=(12, 11))
    for i, g in enumerate(gammas):
        for j, c in enumerate(cs):
            svm = SVC(kernel="rbf", gamma=g, C=c).fit(x, y)
            _boundary(axes[i, j], svm.decision_function, x, y,
                      f"γ = {g:g}, C = {c:g}  (SV {len(svm.support_)})",
                      sv=svm.support_vectors_)
    fig.suptitle("RBF SVM: γ roste shora dolů (dosah vlivu bodu klesá), "
                 "C roste zleva doprava (pásmo tvrdne)")
    _save(fig, "vliv_c_gamma.png")


def fig_gram() -> None:
    """RBF Gramova matice: blokova struktura u oddelitelnych trid."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained")
    for ax, (name, (x, y)) in zip(axes, [
        ("shluky (blobs)", make_blobs(n_samples=120, centers=2, random_state=SEED)),
        ("půlměsíce (moons)", make_moons(n_samples=120, noise=0.15, random_state=SEED)),
    ]):
        x, y = _std(x), _labels(y)
        order = np.argsort(-y, kind="stable")
        xs = x[order]
        k = np.exp(-1.0 * ((xs[:, None, :] - xs[None, :, :]) ** 2).sum(-1))
        im = ax.imshow(k, cmap="Blues", vmin=0, vmax=1)
        n_pos = int((y > 0).sum())
        ax.axhline(n_pos - 0.5, color=POS, lw=1.5)
        ax.axvline(n_pos - 0.5, color=POS, lw=1.5)
        ax.set_title(f"RBF Gramova matice, {name}")
        ax.set_xlabel("body seřazené podle třídy (+1 | −1)")
        ax.set_ylabel("body seřazené podle třídy")
    fig.colorbar(im, ax=axes, label="K[i, j] = podobnost", shrink=0.9)
    _save(fig, "gram_heatmapa.png")


if __name__ == "__main__":
    fig_pasmo()
    fig_soft_margin()
    fig_transformace()
    fig_srovnani()
    fig_vliv_c_gamma()
    fig_gram()
