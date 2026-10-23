"""Datove sady cviceni 12 a jejich rozdeleni se standardizaci (PREDVYPLNENO).

Vsechna data pochazeji ze ``sklearn`` (generatory a vestavena data), nic se
nestahuje a slozka ``data/`` zustava prazdna. Role datasetu:

* **2D hracky** -- ``load_blobs``, ``load_circles``, ``load_moons``: struktura
  je VIDET, slouzi k vizualizaci hranice, pasma a kernelove triku.
* **breast cancer** (30 priznaku) -- realna "slepa" uloha: nakreslit ji
  nelze, jadro a ``c`` se vybiraji validaci (experimentem), ne pohledem.
* ``load_iris_binary`` -- volitelna mala realna uloha (dve tridy kosatcu).

Loadery vraceji SUROVE priznaky (``float64`` tenzor) a tridy ``-1`` / ``+1``.
Standardizace probiha az v ``train_test_split_std`` se statistikami JEN
z trenovaci casti (jinak by informace z testu unikla do predzpracovani,
viz cv6). SVM je citlive na meritko priznaku: jadro i skalarni soucin
se pocitaji ze vzdalenosti a priznak v jinych jednotkach by prevazil ostatni.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import Tensor


def _to_tensors(features: np.ndarray, labels: np.ndarray,
                positive: int) -> tuple[Tensor, Tensor]:
    """Pomocna funkce: priznaky na ``float64`` tenzor, tridy na ``+1`` (positive) / ``-1``."""
    x = torch.as_tensor(np.asarray(features, dtype=np.float64))
    y = torch.as_tensor(np.where(np.asarray(labels) == positive, 1.0, -1.0))
    return x, y


def load_blobs(n_samples: int = 200, cluster_std: float = 1.0,
               seed: int = 42) -> tuple[Tensor, Tensor]:
    """Dva gaussovske shluky v rovine -- linearne oddelitelna data.

    Parametry
    ---------
    n_samples : int, optional
        Celkovy pocet bodu (pul na tridu).
    cluster_std : float, optional
        Smerodatna odchylka shluku; vetsi hodnota = vic prekryvu.
    seed : int, optional
        Seed generatoru.

    Navratova hodnota
    -----------------
    tuple[Tensor, Tensor]
        ``x`` tvaru ``(n, 2)`` a ``y`` v ``{-1, +1}`` tvaru ``(n,)``.
    """
    x, labels = datasets.make_blobs(n_samples=n_samples, centers=2, cluster_std=cluster_std,
                                    random_state=seed)
    return _to_tensors(x, labels, positive=1)


def load_circles(n_samples: int = 200, noise: float = 0.08, factor: float = 0.4,
                 seed: int = 42) -> tuple[Tensor, Tensor]:
    """Dve soustredne kruznice -- primkou neoddelitelne, polynomem st. 2 ano.

    Vnitrni kruznice ma tridu ``+1``, vnejsi ``-1``.

    Parametry
    ---------
    n_samples : int, optional
        Celkovy pocet bodu.
    noise : float, optional
        Smerodatna odchylka gaussovskeho sumu.
    factor : float, optional
        Pomer polomeru vnitrni a vnejsi kruznice, ``0 < factor < 1``.
    seed : int, optional
        Seed generatoru.

    Navratova hodnota
    -----------------
    tuple[Tensor, Tensor]
        ``x`` tvaru ``(n, 2)`` a ``y`` v ``{-1, +1}`` tvaru ``(n,)``.
    """
    x, labels = datasets.make_circles(n_samples=n_samples, noise=noise, factor=factor,
                                      random_state=seed)
    return _to_tensors(x, labels, positive=1)


def load_moons(n_samples: int = 200, noise: float = 0.15,
               seed: int = 42) -> tuple[Tensor, Tensor]:
    """Dva zaklesnute pulmesice -- nelinearni hranice pro RBF jadro.

    Parametry
    ---------
    n_samples : int, optional
        Celkovy pocet bodu.
    noise : float, optional
        Smerodatna odchylka gaussovskeho sumu.
    seed : int, optional
        Seed generatoru.

    Navratova hodnota
    -----------------
    tuple[Tensor, Tensor]
        ``x`` tvaru ``(n, 2)`` a ``y`` v ``{-1, +1}`` tvaru ``(n,)``.
    """
    x, labels = datasets.make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    return _to_tensors(x, labels, positive=1)


def load_breast_cancer() -> tuple[Tensor, Tensor]:
    """Wisconsin Diagnostic Breast Cancer: 569 nalezu, 30 priznaku jader bunek.

    Trida ``+1`` = maligni (212 nalezu), ``-1`` = benigni (357 nalezu).
    Pozor: ``sklearn`` koduje maligni nalez jako 0, zde je to pozitivni trida.

    Navratova hodnota
    -----------------
    tuple[Tensor, Tensor]
        ``x`` tvaru ``(569, 30)`` (surove, nestandardizovane) a ``y`` tvaru ``(569,)``.
    """
    data = datasets.load_breast_cancer()
    return _to_tensors(data.data, data.target, positive=0)


def load_iris_binary() -> tuple[Tensor, Tensor]:
    """Kosatce, jen dve tridy: versicolor (``+1``) a virginica (``-1``), 4 priznaky.

    Tyto dve tridy se castecne prekryvaji (setosa by byla oddelitelna trivialne).

    Navratova hodnota
    -----------------
    tuple[Tensor, Tensor]
        ``x`` tvaru ``(100, 4)`` a ``y`` tvaru ``(100,)``.
    """
    data = datasets.load_iris()
    mask = data.target > 0
    return _to_tensors(data.data[mask], data.target[mask], positive=1)


def train_test_split_std(
    x: Tensor, y: Tensor, test_size: float = 0.3, seed: int = 42,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Stratifikovane nahodne rozdeleni a standardizace statistikami z treninku.

    Parametry
    ---------
    x : Tensor
        Priznaky, tvar ``(n, d)``.
    y : Tensor
        Tridy ``-1`` / ``+1``, tvar ``(n,)``.
    test_size : float, optional
        Podil testovacich bodu, ``0 < test_size < 1``.
    seed : int, optional
        Seed rozdeleni.

    Navratova hodnota
    -----------------
    tuple[Tensor, Tensor, Tensor, Tensor]
        ``x_train, y_train, x_test, y_test`` (``float64``); priznaky maji na
        trenovaci casti nulovy prumer a jednotkovou smerodatnou odchylku.

    Vyjimky
    -------
    ``ValueError``:
        Pokud ``test_size`` nelezi v ``(0, 1)`` nebo nesouhlasi delky ``x`` a ``y``.
    """
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size musi lezet v intervalu (0, 1), zadano: {test_size}")
    x_np = np.asarray(x, dtype=np.float64)
    y_np = np.asarray(y, dtype=np.float64)
    if x_np.ndim != 2 or y_np.ndim != 1 or x_np.shape[0] != y_np.shape[0]:
        raise ValueError(
            f"ocekavano x tvaru (n, d) a y tvaru (n,), zadano {x_np.shape} a {y_np.shape}"
        )
    x_tr, x_te, y_tr, y_te = train_test_split(
        x_np, y_np, test_size=test_size, random_state=seed, stratify=y_np,
    )
    scaler = StandardScaler().fit(x_tr)
    as_t = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float64)  # noqa: E731
    return as_t(scaler.transform(x_tr)), as_t(y_tr), as_t(scaler.transform(x_te)), as_t(y_te)
