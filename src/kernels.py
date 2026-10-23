"""Kernelova knihovna: jadra jako MATICE PODOBNOSTI (Gramovy matice).

Jadro (kernel) je funkce dvou bodu ``kappa(a, b)``, ktera vraci jejich
podobnost. Matematicky jde o skalarni soucin obrazu bodu v jinem
(prizpusobenem) prostoru, ``kappa(a, b) = phi(a) . phi(b)``. Kernelovy trik
spociva v tom, ze tento soucin pocitame PRIMO z puvodnich bodu ``a``, ``b``,
aniz bychom zobrazeni ``phi`` kdy sestavili.

Knihovna proto NEVRACI priznaky, ale cisla podobnosti: zakladni operaci je
``gram(x, z)``, ktera pro dve sady bodu vrati matici ``K`` tvaru
``(x.shape[0], z.shape[0])`` s prvky ``K[i, j] = kappa(x[i], z[j])``.

Vzor Strategy + Factory (jako ``Initializer`` v cv3): jadra jsou zamenitelne
strategie se spolecnym rozhranim ``Kernel.gram`` a tovarni funkce
``make_kernel`` vybere strategii podle retezce z ``config.yaml``.

Jadro **implementujete** (jak se pocita), ale **vybirate** ho experimentem
(validaci na datech), ne odvozenim.

Vsechny tenzory jsou typu ``torch.float64``, radek = jeden bod.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor


class Kernel(ABC):
    """Spolecne rozhrani vsech jader (PREDVYPLNENO).

    Kazde jadro umi jedine: spocitat matici podobnosti mezi dvema sadami
    bodu. Treninkovy algoritmus (``KernelSVM``) pak nevi a nemusi vedet,
    ktere jadro dostal -- vidi jen matici ``K``.

    Atribut tridy ``name`` je jmeno jadra v registru ``make_kernel``
    (a v ``config.yaml``); pouziva se pri ukladani modelu.
    """

    name: str = ""

    @abstractmethod
    def gram(self, x: Tensor, z: Tensor) -> Tensor:
        """Matice podobnosti ``K``, ``K[i, j] = kappa(x[i], z[j])``.

        Parametry
        ---------
        x : Tensor
            Prvni sada bodu, tvar ``(n, d)``.
        z : Tensor
            Druha sada bodu, tvar ``(m, d)`` (stejny pocet priznaku ``d``).

        Navratova hodnota
        -----------------
        Tensor
            Matice podobnosti tvaru ``(n, m)``, typ ``float64``.
        """

    def __repr__(self) -> str:
        """Citelny zapis jadra i s parametry, napr. ``RBFKernel(gamma=1.0)``."""
        params = ", ".join(f"{k}={v!r}" for k, v in vars(self).items())
        return f"{type(self).__name__}({params})"


class LinearKernel(Kernel):
    """Linearni jadro ``kappa(a, b) = a^T b`` (VZOR, predvyplneno).

    Odpovida ``phi(x) = x`` -- zadne zobrazeni, puvodni prostor. SVM s timto
    jadrem hleda primku (nadrovinu) v puvodnim prostoru, stejne jako
    perceptron z cv8, jen tu s nejsirsim pasmem.

    Pouzijte tuto tridu jako vzor pro ``PolynomialKernel`` a ``RBFKernel``:
    kontrola vstupu, jeden radek maticoveho vypoctu, vraceni matice.
    """

    name = "linear"

    def gram(self, x: Tensor, z: Tensor) -> Tensor:
        """Matice skalarnich soucinu ``K = x @ z.T`` (VZOR).

        Parametry
        ---------
        x : Tensor
            Tvar ``(n, d)``.
        z : Tensor
            Tvar ``(m, d)``.

        Navratova hodnota
        -----------------
        Tensor
            ``K[i, j] = x[i] . z[j]``, tvar ``(n, m)``.
        """
        assert x.ndim == 2 and z.ndim == 2, "x i z musi byt matice (n, d)"
        assert x.shape[1] == z.shape[1], "x a z musi mit stejny pocet priznaku"
        return x @ z.T


class PolynomialKernel(Kernel):
    """Polynomialni (homogenni) jadro ``kappa(a, b) = (a^T b)^r`` (UKOL).

    Pro ``r = 2`` a dvourozmerna data lze zobrazeni ``phi`` vypsat:
    ``phi(x) = (x1^2, sqrt(2) x1 x2, x2^2)`` a plati
    ``phi(a) . phi(b) = (a^T b)^2``. Na tom si kernelovy trik overite
    vlastnima ocima (``feature_map``). Pro vyssi ``r`` a vic priznaku by
    ``phi`` mela neunosne mnoho slozek -- proto trik.

    Atributy
    --------
    r : int
        Stupen polynomu, ``r >= 1``.
    """

    name = "poly"

    def __init__(self, r: int = 2) -> None:
        """Ulozi stupen polynomu (PREDVYPLNENO).

        Parametry
        ---------
        r : int, optional
            Stupen polynomu (vychozi 2), cele cislo ``>= 1``.
        """
        if isinstance(r, bool) or not isinstance(r, int) or r < 1:
            raise ValueError(f"stupen polynomu r musi byt cele cislo >= 1, zadano: {r!r}")
        self.r = r

    def gram(self, x: Tensor, z: Tensor) -> Tensor:
        """Matice polynomialnich podobnosti ``K = (x @ z.T) ** r`` (UKOL).

        Postup:

        1. Overte vstupy stejne jako ``LinearKernel.gram``.
        2. Spoctete matici skalarnich soucinu ``x @ z.T`` (tvar ``(n, m)``).
        3. Kazdy prvek umocnete na ``self.r`` (po prvcich, ne maticova mocnina!).

        Parametry
        ---------
        x : Tensor
            Tvar ``(n, d)``.
        z : Tensor
            Tvar ``(m, d)``.

        Navratova hodnota
        -----------------
        Tensor
            ``K[i, j] = (x[i] . z[j]) ** r``, tvar ``(n, m)``.
        """
        # assert  Ověřte, že x i z jsou matice (ndim == 2)
        # assert  Ověřte, že x a z mají stejný počet příznaků (sloupců)
        raise NotImplementedError(
            "Úkol: doplnte PolynomialKernel.gram -- matice polynomialnich podobnosti "
            "(a^T b)^r vsech dvojic bodu x a z."
        )

    def feature_map(self, x: Tensor) -> Tensor:
        """Explicitni zobrazeni ``phi`` pro ``r = 2`` a 2D data (UKOL).

        Slouzi JEN k overeni kernelove triku: ``gram(x, z)`` se musi rovnat
        ``feature_map(x) @ feature_map(z).T``. SVM tuto metodu nepouziva.

        Pro bod ``x = (x1, x2)`` vratte radek ``(x1^2, sqrt(2) * x1 * x2, x2^2)``.
        Vsechny radky naraz: sloupce ``x[:, 0]`` a ``x[:, 1]`` zkombinujte po
        prvcich a slozte vedle sebe (``torch.stack(..., dim=1)``).

        Parametry
        ---------
        x : Tensor
            Body, tvar ``(n, 2)``.

        Navratova hodnota
        -----------------
        Tensor
            Obrazy bodu ve 3D prostoru priznaku, tvar ``(n, 3)``.

        Vyjimky
        -------
        ``ValueError``:
            Pokud ``self.r != 2`` (pro jiny stupen tento vzorec neplati).
        """
        if self.r != 2:
            raise ValueError(f"feature_map je definovana jen pro r = 2, zadano r = {self.r}")
        # assert  Ověřte, že x má tvar (n, 2)
        raise NotImplementedError(
            "Úkol: doplnte PolynomialKernel.feature_map -- explicitni zobrazeni phi "
            "2D bodu do 3D prostoru priznaku polynomu stupne 2."
        )


class RBFKernel(Kernel):
    """Gaussovske (RBF) jadro ``kappa(a, b) = exp(-gamma * ||a - b||^2)`` (UKOL).

    ``gamma = 1 / (2 sigma^2)`` urcuje dosah vlivu jednoho bodu: male
    ``gamma`` = siroky vliv, hladka hranice; velke ``gamma`` = vliv jen na
    nejblizsi okoli, hranice obtaci jednotlive body (preuceni).

    Proc tu NENI ``feature_map``: zobrazeni ``phi`` jadra RBF ma nekonecne
    mnoho slozek (rozvoj ``exp`` v Taylorovu radu obsahuje polynomy vsech
    stupnu). Vypsat ho nelze -- kernelovy trik je tu nutnost, ne zkratka.

    Atributy
    --------
    gamma : float
        Sirka jadra, ``gamma > 0``.
    """

    name = "rbf"

    def __init__(self, gamma: float) -> None:
        """Ulozi parametr ``gamma`` (PREDVYPLNENO).

        Parametry
        ---------
        gamma : float
            Kladne konecne cislo.
        """
        if isinstance(gamma, bool) or not isinstance(gamma, (int, float)) \
                or not 0 < gamma < float("inf"):
            raise ValueError(f"gamma musi byt kladne konecne cislo, zadano: {gamma!r}")
        self.gamma = float(gamma)

    def gram(self, x: Tensor, z: Tensor) -> Tensor:
        """Matice RBF podobnosti ``K[i, j] = exp(-gamma * ||x[i] - z[j]||^2)`` (UKOL).

        Ctverce vzdalenosti vsech dvojic spoctete BEZ smycek z rozkladu

            ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a . b

        Postup:

        1. Overte vstupy stejne jako ``LinearKernel.gram``.
        2. ``x_sq`` = soucet ctvercu kazdeho radku ``x``, tvar ``(n,)``;
           ``z_sq`` obdobne pro ``z``, tvar ``(m,)``.
        3. Matice ctvercu vzdalenosti ``(n, m)``: ``x_sq`` jako sloupec
           (``x_sq[:, None]``) + ``z_sq`` jako radek (``z_sq[None, :]``)
           - ``2 * x @ z.T`` (broadcasting).
        4. Zaokrouhlovaci chyby mohou dat male zaporne cislo (napr. ``-1e-16``)
           tam, kde ma byt 0 -- oriznete zdola nulou (``clamp_min(0.0)``).
        5. Vratte ``exp(-self.gamma * vzdalenosti)``.

        Kontrola: ``gram(x, x)`` je symetricka a ma na diagonale jednicky
        (kazdy bod je sam sobe maximalne podobny).

        Parametry
        ---------
        x : Tensor
            Tvar ``(n, d)``.
        z : Tensor
            Tvar ``(m, d)``.

        Navratova hodnota
        -----------------
        Tensor
            Matice podobnosti s hodnotami v ``(0, 1]``, tvar ``(n, m)``.
        """
        # assert  Ověřte, že x i z jsou matice (ndim == 2)
        # assert  Ověřte, že x a z mají stejný počet příznaků (sloupců)
        raise NotImplementedError(
            "Úkol: doplnte RBFKernel.gram -- matice RBF podobnosti vsech dvojic bodu "
            "bez smycek (postup v docstringu)."
        )


def make_kernel(name: str, **params: object) -> Kernel:
    """Tovarni funkce: vytvori jadro podle jmena z ``config.yaml`` (UKOL).

    Registr jmen (slovnik ``jmeno -> trida``):

    ========== ==================== ==========================
    jmeno      trida                pouzite parametry
    ========== ==================== ==========================
    ``linear`` ``LinearKernel``     zadne
    ``poly``   ``PolynomialKernel`` ``r``
    ``rbf``    ``RBFKernel``        ``gamma``
    ========== ==================== ==========================

    z ``params`` predejte konstruktoru JEN parametry, ktere dana trida
    pouziva (konfigurace predava ``r`` i ``gamma`` vzdy, at je zvolene
    jakekoli jadro). Napr. ``make_kernel("rbf", r=2, gamma=0.5)`` vrati
    ``RBFKernel(gamma=0.5)``.

    Parametry
    ---------
    name : str
        Jmeno jadra: ``"linear"``, ``"poly"`` nebo ``"rbf"``.
    **params
        Parametry jader (``r``, ``gamma``); nepotrebne se ignoruji.

    Navratova hodnota
    -----------------
    Kernel
        Nova instance zvoleneho jadra.

    Vyjimky
    -------
    ``ValueError``:
        Pokud ``name`` neni v registru (hlaska vyjmenuje povolena jmena).
    """
    raise NotImplementedError(
        "Úkol: doplnte make_kernel -- tovarni funkce, ktera podle jmena z config.yaml "
        "vytvori odpovidajici jadro (registr a parametry v docstringu)."
    )


def verify_kernel_trick(kernel: PolynomialKernel, x: Tensor, z: Tensor) -> float:
    """Overeni kernelove triku ``K = phi . phi`` na vlastni oci (UKOL, lehky).

    Pseudokod:

    * over, ze jde o polynom stupne 2
    * matice A: podobnosti kernelovym trikem (jadro primo v puvodnim 2D prostoru)
    * matice B: obe sady bodu zobraz do 3D prostoru priznaku a spocti skalarni
      soucin kazde dvojice obrazu
    * vrat nejvetsi absolutni rozdil prvku A a B jako python ``float``

    Spravne implementovane jadro da rozdil na urovni zaokrouhlovacich chyb
    (radove ``1e-15``).

    Parametry
    ---------
    kernel : PolynomialKernel
        Polynomialni jadro stupne ``r = 2``.
    x : Tensor
        Tvar ``(n, 2)``.
    z : Tensor
        Tvar ``(m, 2)``.

    Navratova hodnota
    -----------------
    float
        Nejvetsi absolutni rozdil obou matic tvaru ``(n, m)``.
    """
    # assert  Ověřte, že kernel.r == 2
    raise NotImplementedError(
        "Úkol: doplnte verify_kernel_trick -- porovnejte matici jadra s matici "
        "skalarnich soucinu v prostoru priznaku phi."
    )
