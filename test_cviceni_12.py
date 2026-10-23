# -*- coding: utf-8 -*-

"""
Created on 14. 10. 2026 at 10:20:00

Author: Richard Redina
Email: 195715@vut.cz
Affiliation:
         International Clinical Research Center, Brno
         Brno University of Technology, Brno
GitHub: RicRedi

(._.)
 <|>
_/|_

Description:
    Testy pro cviceni 12 -- SVM a kernelovy trik (dual, PyTorch).

    Spousteni:  pytest -v

    Ve stavu stubu se sada NACTE a testy, ktere volaji nedokoncene ukoly,
    se oznaci jako xfail (ocekavane selhani s NotImplementedError) -- sada
    nikdy neskonci holym tracebackem. Testy predvyplnenych casti (LinearKernel,
    predvyplnene casti KernelSVM, dataio, utils) projdou v kazdem stavu.

    Testy nikdy nezakladaji soubory v logs/ (logger se hned po importech
    nakonfiguruje bez souboru) a modely i grafy ukladaji jen do pytest
    ``tmp_path``. Testy SVM pouzivaji hotove ``LinearKernel`` (VZOR), kde to
    jde, takze nezavisi na studentovych ``PolynomialKernel`` a ``RBFKernel``.
    Referencni hodnoty (oracle) se pocitaji primo v testech rucnimi vzorci
    z NumPy / sklearn, nikdy studentskym kodem.

    Deset trid testu:
      TestLinearKernel     -- hotovy vzor jadra
      TestPolynomialKernel -- (a.b)^r, feature_map a kernelovy trik
      TestRBFKernel        -- exp(-gamma ||a-b||^2): hodnoty, symetrie, diagonala, PSD
      TestMakeKernel       -- tovarni funkce a registr jmen
      TestDual             -- dualni ucelova funkce a krok gradientniho vystupu (jadro)
      TestKernelSVM        -- predvyplnene casti, support vektory, rozhodovaci funkce,
                              trenink na blobs a kruznicich
      TestSaveLoad         -- slovnik do .pt a zpet
      TestSweepC           -- vliv c: jeden radek na c, klice, sirka pasma
      TestDataAKonfigurace -- loadery, rozdeleni se standardizaci, config.yaml
      TestPipelineUtils    -- utils/: kontrola treninku, prepravky, hlasky (bez studenta)
================================================================================
"""

from __future__ import annotations

import copy
import dataclasses
import math
import os

import numpy as np
import pytest
import torch
from sklearn import metrics as skm
from sklearn.metrics.pairwise import polynomial_kernel, rbf_kernel

from dataio import (
    ExperimentConfig,
    load_blobs,
    load_breast_cancer,
    load_circles,
    load_config,
    load_iris_binary,
    load_moons,
    plot_boundary_panels,
    plot_c_sweep,
    plot_decision_boundary,
    plot_dual_history,
    plot_gram_heatmap,
    train_test_split_std,
    validate_config,
)
from src import (
    Kernel,
    KernelSVM,
    LinearKernel,
    PolynomialKernel,
    RBFKernel,
    make_kernel,
    sweep_c,
    verify_kernel_trick,
)
from src.svm import get_logger
from utils import (
    DatasetBundle,
    SweepRow,
    UceniDivergovalo,
    chyba_dat,
    faze_neni_hotova,
    kontrola_uceni,
    nauc_perceptron,
    preskoceno,
    radky_sweepu,
    rozdel,
    tovarna_modelu,
    vyhodnot_klasifikaci,
    vypis_metriky,
)
from utils import confusion_matrix as cm_utils

get_logger(None)   # logger jen do konzole: testy nikdy nezalozi soubor v logs/

STUB = pytest.mark.xfail(raises=NotImplementedError, strict=False,
                         reason="studentsky ukol jeste neni dokoncen")

ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_YAML = os.path.join(ROOT, "config.yaml")


# --------------------------------------------------------------------------- #
#  Pomocne funkce                                                             #
# --------------------------------------------------------------------------- #
def _points(n: int, d: int, seed: int) -> torch.Tensor:
    """Nahodne body ``(n, d)`` typu ``float64``."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=g, dtype=torch.float64)


def _toy() -> tuple[torch.Tensor, torch.Tensor]:
    """Ctyri body ve 2D, dve tridy, linearne oddelitelne."""
    x = torch.tensor([[2.0, 2.0], [1.0, 3.0], [-1.0, -2.0], [-2.0, -1.0]], dtype=torch.float64)
    y = torch.tensor([1.0, 1.0, -1.0, -1.0], dtype=torch.float64)
    return x, y


def _oracle_step(lam: np.ndarray, k_eff: np.ndarray, y: np.ndarray, lr: float,
                 c: float) -> np.ndarray:
    """Rucni krok projektovaneho gradientniho vystupu: clip(lam + lr (1 - q lam), 0, c)."""
    q = np.outer(y, y) * k_eff
    return np.clip(lam + lr * (1.0 - q @ lam), 0.0, c)


def _bundle(x: torch.Tensor, y: torch.Tensor, name: str = "test") -> DatasetBundle:
    """Rozdeli data do ``DatasetBundle`` (30 % test, seed 0)."""
    return rozdel(x, y, name, test_size=0.3, seed=0)


def _blobs_bundle() -> DatasetBundle:
    """Dobre oddelitelne shluky, 120 bodu."""
    x, y = load_blobs(n_samples=120, cluster_std=1.0, seed=42)
    return _bundle(x, y, "blobs")


# --------------------------------------------------------------------------- #
#  Jadra                                                                      #
# --------------------------------------------------------------------------- #
class TestLinearKernel:
    """Hotovy vzor ``LinearKernel``."""

    def test_hodnoty_a_tvar(self) -> None:
        """``K = x z^T``, tvar ``(n, m)``."""
        x, z = _points(5, 3, 0), _points(7, 3, 1)
        k = LinearKernel().gram(x, z)
        assert k.shape == (5, 7)
        assert torch.allclose(k, x @ z.T)

    def test_ruzny_pocet_priznaku_je_chyba(self) -> None:
        """``x`` a ``z`` s ruznym poctem sloupcu vyhodi ``AssertionError``."""
        with pytest.raises(AssertionError):
            LinearKernel().gram(_points(3, 2, 0), _points(3, 3, 1))

    def test_je_kernel_a_ma_jmeno(self) -> None:
        """Potomek ``Kernel`` s ``name == "linear"``."""
        assert isinstance(LinearKernel(), Kernel)
        assert LinearKernel.name == "linear"


class TestPolynomialKernel:
    """``PolynomialKernel``: ``(a^T b)^r`` a explicitni ``phi`` pro ``r = 2``."""

    @STUB
    @pytest.mark.parametrize("r", [1, 2, 3])
    def test_proti_sklearn(self, r: int) -> None:
        """Shoda se ``sklearn`` (``gamma = 1``, ``coef0 = 0``) a tvar ``(n, m)``."""
        x, z = _points(6, 2, 0), _points(4, 2, 1)
        k = PolynomialKernel(r).gram(x, z)
        assert k.shape == (6, 4)
        ref = polynomial_kernel(x.numpy(), z.numpy(), degree=r, gamma=1.0, coef0=0.0)
        assert np.allclose(k.numpy(), ref)

    @STUB
    def test_mocnina_po_prvcich(self) -> None:
        """Umocnuje se po prvcich, ne maticove (ctvercova matice by to prozradila)."""
        x = torch.tensor([[1.0, 2.0], [3.0, 1.0]], dtype=torch.float64)
        k = PolynomialKernel(2).gram(x, x)
        assert torch.allclose(k, torch.tensor([[25.0, 25.0], [25.0, 100.0]],
                                              dtype=torch.float64))

    @STUB
    def test_feature_map_hodnoty(self) -> None:
        """``phi(x) = (x1^2, sqrt2 x1 x2, x2^2)``, tvar ``(n, 3)``."""
        x = torch.tensor([[1.0, 2.0], [-3.0, 0.5]], dtype=torch.float64)
        phi = PolynomialKernel(2).feature_map(x)
        s2 = math.sqrt(2.0)
        ocekavano = torch.tensor([[1.0, 2.0 * s2, 4.0], [9.0, -1.5 * s2, 0.25]],
                                 dtype=torch.float64)
        assert phi.shape == (2, 3)
        assert torch.allclose(phi, ocekavano)

    @STUB
    def test_kernelovy_trik(self) -> None:
        """``gram(x, z) == phi(x) phi(z)^T`` na urovni zaokrouhlovacich chyb."""
        kernel = PolynomialKernel(2)
        x, z = _points(20, 2, 3), _points(15, 2, 4)
        rozdil = (kernel.gram(x, z) - kernel.feature_map(x) @ kernel.feature_map(z).T)
        assert float(rozdil.abs().max()) < 1e-12

    @STUB
    def test_verify_kernel_trick(self) -> None:
        """``verify_kernel_trick`` vraci python ``float`` blizky nule."""
        rozdil = verify_kernel_trick(PolynomialKernel(2), _points(10, 2, 0), _points(8, 2, 1))
        assert isinstance(rozdil, float)
        assert 0.0 <= rozdil < 1e-12

    def test_feature_map_jen_pro_r2(self) -> None:
        """Pro ``r != 2`` je ``feature_map`` chyba (predvyplnena kontrola)."""
        with pytest.raises(ValueError):
            PolynomialKernel(3).feature_map(_points(3, 2, 0))

    def test_neplatne_r(self) -> None:
        """``r < 1`` nebo necele ``r`` vyhodi ``ValueError``."""
        for spatne in (0, -1, 1.5, True):
            with pytest.raises(ValueError):
                PolynomialKernel(spatne)


class TestRBFKernel:
    """``RBFKernel``: ``exp(-gamma ||a - b||^2)``."""

    @STUB
    @pytest.mark.parametrize("gamma", [0.1, 1.0, 5.0])
    def test_proti_sklearn(self, gamma: float) -> None:
        """Shoda se ``sklearn.metrics.pairwise.rbf_kernel``, tvar ``(n, m)``."""
        x, z = _points(7, 3, 0), _points(5, 3, 1)
        k = RBFKernel(gamma).gram(x, z)
        assert k.shape == (7, 5)
        assert np.allclose(k.numpy(), rbf_kernel(x.numpy(), z.numpy(), gamma=gamma))

    @STUB
    def test_symetrie_diagonala_rozsah(self) -> None:
        """``K(x, x)`` je symetricka, diagonala presne 1, hodnoty v ``(0, 1]``."""
        x = _points(30, 2, 2)
        k = RBFKernel(1.0).gram(x, x)
        assert torch.allclose(k, k.T)
        assert torch.allclose(torch.diagonal(k), torch.ones(30, dtype=torch.float64))
        assert float(k.min()) > 0.0 and float(k.max()) <= 1.0

    @STUB
    def test_psd(self) -> None:
        """Gramova matice je pozitivne semidefinitni (Mercer): vlastni cisla >= 0."""
        x = _points(40, 2, 5)
        eig = torch.linalg.eigvalsh(RBFKernel(0.5).gram(x, x))
        assert float(eig.min()) > -1e-10

    @STUB
    def test_bez_zapornych_vzdalenosti(self) -> None:
        """Body daleko od pocatku: bez orezu nulou vyjde ||a-a||^2 < 0 a K > 1."""
        g = torch.Generator().manual_seed(0)
        x = torch.randn(30, 5, generator=g, dtype=torch.float64) * 1e3 + 5e3
        x_sq = (x ** 2).sum(dim=1)
        bez_orezu = x_sq[:, None] + x_sq[None, :] - 2.0 * x @ x.T
        assert float(bez_orezu.diagonal().min()) < 0.0      # data skutecne past obsahuji
        k = RBFKernel(1.0).gram(x, x)
        assert float(k.max()) <= 1.0
        assert torch.allclose(torch.diagonal(k), torch.ones(30, dtype=torch.float64))

    def test_neplatne_gamma(self) -> None:
        """``gamma <= 0`` vyhodi ``ValueError`` (predvyplnena kontrola)."""
        for spatne in (0.0, -1.0, float("nan"), float("inf")):
            with pytest.raises(ValueError):
                RBFKernel(spatne)


class TestMakeKernel:
    """Tovarni funkce ``make_kernel`` (Strategy + Factory)."""

    @STUB
    def test_typy_a_parametry(self) -> None:
        """Jmeno vybere tridu, parametry se predaji jen te, ktera je pouziva."""
        lin = make_kernel("linear", r=3, gamma=2.0)
        poly = make_kernel("poly", r=3, gamma=2.0)
        rbf = make_kernel("rbf", r=3, gamma=2.0)
        assert type(lin) is LinearKernel
        assert type(poly) is PolynomialKernel and poly.r == 3
        assert type(rbf) is RBFKernel and rbf.gamma == 2.0

    @STUB
    def test_nezname_jmeno(self) -> None:
        """Nezname jmeno vyhodi ``ValueError``."""
        with pytest.raises(ValueError):
            make_kernel("sigmoid", r=2, gamma=1.0)

    @STUB
    def test_nova_instance(self) -> None:
        """Kazde volani vraci novou instanci."""
        assert make_kernel("rbf", gamma=1.0) is not make_kernel("rbf", gamma=1.0)


# --------------------------------------------------------------------------- #
#  Dual (jadro cviceni)                                                       #
# --------------------------------------------------------------------------- #
class TestDual:
    """``dual_objective`` a ``_ascent_step``."""

    def _svm(self, c: float = 1.0, lr: float = 0.01) -> KernelSVM:
        return KernelSVM(LinearKernel(), c=c, lr=lr, n_iters=10)

    @STUB
    def test_dual_objective_hodnota(self) -> None:
        """Hodnota ``L`` souhlasi s rucni dvojitou sumou."""
        x, y = _toy()
        svm = self._svm()
        k_eff = svm._effective_gram(x, x)
        lam = torch.tensor([0.1, 0.0, 0.3, 0.2], dtype=torch.float64)
        dual = svm.dual_objective(lam, k_eff, y)
        ocekavano = float(lam.sum()) - 0.5 * sum(
            float(lam[i] * lam[j] * y[i] * y[j] * k_eff[i, j])
            for i in range(4) for j in range(4)
        )
        assert isinstance(dual, torch.Tensor) and dual.ndim == 0
        assert math.isclose(float(dual), ocekavano, rel_tol=1e-12)

    @STUB
    def test_dual_objective_v_nule(self) -> None:
        """``L(0) = 0``."""
        x, y = _toy()
        svm = self._svm()
        dual = svm.dual_objective(torch.zeros(4, dtype=torch.float64),
                                  svm._effective_gram(x, x), y)
        assert float(dual) == 0.0

    @STUB
    def test_dual_objective_drzi_graf(self) -> None:
        """z vysledku lze autogradem spocitat gradient ``1 - q lam``."""
        x, y = _toy()
        svm = self._svm()
        k_eff = svm._effective_gram(x, x)
        lam = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64, requires_grad=True)
        svm.dual_objective(lam, k_eff, y).backward()
        q = torch.outer(y, y) * k_eff
        assert torch.allclose(lam.grad, 1.0 - q @ lam.detach())

    @STUB
    def test_krok_proti_rucnimu_vzorci(self) -> None:
        """Krok = ``clip(lam + lr (1 - q lam), 0, c)`` (vystup PO gradientu + projekce)."""
        x, y = _toy()
        svm = self._svm(c=0.5, lr=0.05)
        k_eff = svm._effective_gram(x, x)
        lam = torch.tensor([0.0, 0.45, 0.2, 0.1], dtype=torch.float64)
        nove = svm._ascent_step(lam, k_eff, y)
        ocekavano = _oracle_step(lam.numpy(), k_eff.numpy(), y.numpy(), 0.05, 0.5)
        assert nove.shape == (4,)
        assert np.allclose(nove.numpy(), ocekavano, atol=1e-12)

    @STUB
    def test_krok_v_boxu_a_bez_grafu(self) -> None:
        """Vysledek lezi v ``[0, c]``, nema ``requires_grad`` a nemeni vstupni ``lam``."""
        x, y = _toy()
        svm = self._svm(c=0.2, lr=1.0)
        k_eff = svm._effective_gram(x, x)
        lam = torch.tensor([0.1, 0.1, 0.1, 0.1], dtype=torch.float64)
        puvodni = lam.clone()
        nove = svm._ascent_step(lam, k_eff, y)
        assert not nove.requires_grad
        assert float(nove.min()) >= 0.0 and float(nove.max()) <= 0.2
        assert torch.equal(lam, puvodni)

    @STUB
    def test_opakovane_kroky_zvysuji_dual(self) -> None:
        """Pri malem kroku ``L(lambda)`` neklesa (maximalizujeme)."""
        x, y = _toy()
        svm = self._svm(c=1.0, lr=0.01)
        k_eff = svm._effective_gram(x, x)
        lam = torch.zeros(4, dtype=torch.float64)
        hodnoty = []
        for _ in range(50):
            lam = svm._ascent_step(lam, k_eff, y)
            q = torch.outer(y, y) * k_eff
            hodnoty.append(float(lam.sum() - 0.5 * lam @ q @ lam))
        assert all(b >= a - 1e-12 for a, b in zip(hodnoty, hodnoty[1:]))
        assert hodnoty[-1] > 0.0


# --------------------------------------------------------------------------- #
#  KernelSVM                                                                  #
# --------------------------------------------------------------------------- #
class TestKernelSVM:
    """Predvyplnene casti ``KernelSVM`` a studentske vybavovani."""

    def test_konstruktor_validuje(self) -> None:
        """Spatne jadro nebo hyperparametry vyhodi chybu."""
        with pytest.raises(TypeError):
            KernelSVM("rbf", c=1.0, lr=0.1, n_iters=10)
        for kwargs in ({"c": 0.0}, {"lr": -1.0}, {"n_iters": 0}, {"n_iters": 2.5}):
            parametry = {"c": 1.0, "lr": 0.1, "n_iters": 10} | kwargs
            with pytest.raises(ValueError):
                KernelSVM(LinearKernel(), **parametry)

    def test_effective_gram_pricita_jednicku(self) -> None:
        """``k_eff = K + 1`` (bias foldnuty do jadra)."""
        x, z = _points(4, 2, 0), _points(3, 2, 1)
        svm = KernelSVM(LinearKernel(), c=1.0, lr=0.1, n_iters=1)
        assert torch.allclose(svm._effective_gram(x, z), x @ z.T + 1.0)

    def test_fit_odmitne_spatne_tridy(self) -> None:
        """Tridy jine nez ``-1`` / ``+1`` (napr. 0/1) jsou chyba hned na zacatku."""
        x, _ = _toy()
        with pytest.raises(ValueError):
            KernelSVM(LinearKernel(), c=1.0, lr=0.1, n_iters=5).fit(
                x, torch.tensor([1.0, 1.0, 0.0, 0.0]))

    @staticmethod
    def _bez_studenta(svm: KernelSVM, monkeypatch: pytest.MonkeyPatch) -> None:
        """Nahradi studentske metody, aby slo testovat jen predvyplnenou smycku ``fit``."""
        monkeypatch.setattr(svm, "_ascent_step", lambda lam, k, yy: lam)
        monkeypatch.setattr(svm, "dual_objective",
                            lambda lam, k, yy: torch.tensor(0.0, dtype=torch.float64))
        monkeypatch.setattr(svm, "support_vector_indices",
                            lambda: torch.zeros(0, dtype=torch.int64))

    def test_automaticky_krok(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Pri ``lr=None`` urci ``fit`` krok ``1 / mu_max`` matice ``q = outer(y, y) * k_eff``."""
        x, y = _toy()
        svm = KernelSVM(LinearKernel(), c=1.0, lr=None, n_iters=3)
        self._bez_studenta(svm, monkeypatch)
        svm.fit(x, y)
        q = torch.outer(y, y) * (x @ x.T + 1.0)
        assert math.isclose(svm.step_, 1.0 / float(torch.linalg.eigvalsh(q).max()))

    def test_rucni_krok(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Zadane ``lr`` se pouzije beze zmeny."""
        x, y = _toy()
        svm = KernelSVM(LinearKernel(), c=1.0, lr=0.02, n_iters=3)
        self._bez_studenta(svm, monkeypatch)
        assert svm.fit(x, y).step_ == 0.02

    def test_historie_ve_stovce_bodu(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Historie se zapisuje v 1. iteraci a po kazde setine iteraci (101 zaznamu)."""
        x, y = _toy()
        svm = KernelSVM(LinearKernel(), c=1.0, lr=0.01, n_iters=1000)
        self._bez_studenta(svm, monkeypatch)
        iterace = svm.fit(x, y).history_["iteration"]
        assert len(iterace) == 101
        assert iterace[:3] == [1, 10, 20] and iterace[-1] == 1000

    def test_fit_odhali_chybejici_projekci(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Krok, ktery opusti box ``[0, c]``, ``fit`` ohlasi srozumitelnou chybou."""
        x, y = _toy()
        svm = KernelSVM(LinearKernel(), c=1.0, lr=0.1, n_iters=5)
        monkeypatch.setattr(svm, "_ascent_step", lambda lam, k, yy: lam + 5.0)
        with pytest.raises(ValueError, match="box"):
            svm.fit(x, y)

    def test_predict_podle_znamenka(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``predict`` vraci ``+1`` pro ``h >= 0`` a ``-1`` jinak."""
        svm = KernelSVM(LinearKernel(), c=1.0, lr=0.1, n_iters=1)
        monkeypatch.setattr(svm, "decision_function",
                            lambda x: torch.tensor([2.0, 0.0, -0.1], dtype=torch.float64))
        assert torch.equal(svm.predict(_points(3, 2, 0)),
                           torch.tensor([1.0, 1.0, -1.0], dtype=torch.float64))

    @STUB
    def test_support_vector_indices(self) -> None:
        """Indexy, kde ``lambda > sv_threshold``, typ ``int64``, vzestupne."""
        svm = KernelSVM(LinearKernel(), c=1.0, lr=0.1, n_iters=1, sv_threshold=1e-3)
        svm.lambdas_ = torch.tensor([0.0, 0.5, 1e-4, 1.0, 2e-3], dtype=torch.float64)
        idx = svm.support_vector_indices()
        assert idx.dtype == torch.int64
        assert idx.tolist() == [1, 3, 4]

    @STUB
    def test_decision_function_vzorec(self) -> None:
        """``h(x) = sum lambda_i y_i (K(x_i, x) + 1)`` = ``sum lambda_i y_i K + b``."""
        svm = KernelSVM(LinearKernel(), c=1.0, lr=0.1, n_iters=1)
        svm.sv_x_ = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
        svm.sv_y_ = torch.tensor([1.0, -1.0], dtype=torch.float64)
        svm.sv_lambdas_ = torch.tensor([0.5, 0.25], dtype=torch.float64)
        x = torch.tensor([[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]], dtype=torch.float64)
        h = svm.decision_function(x)
        # 0.5*(x1 + 1) - 0.25*(x2 + 1)
        assert h.shape == (3,)
        assert torch.allclose(h, torch.tensor([1.25, -0.25, 0.5], dtype=torch.float64))

    @STUB
    def test_fit_shoda_s_rucnim_vypoctem(self) -> None:
        """``lambdas_`` po ``fit`` = opakovany rucni krok; SV, ``bias_`` a historie sedi."""
        x, y = _toy()
        svm = KernelSVM(LinearKernel(), c=0.3, lr=0.02, n_iters=40).fit(x, y)
        k_eff = (x @ x.T + 1.0).numpy()
        lam = np.zeros(4)
        for _ in range(40):
            lam = _oracle_step(lam, k_eff, y.numpy(), 0.02, 0.3)
        assert np.allclose(svm.lambdas_.numpy(), lam, atol=1e-12)
        sv = np.nonzero(lam > 1e-5)[0]
        assert svm.support_.tolist() == sv.tolist()
        assert math.isclose(svm.bias_, float((lam * y.numpy()).sum()), abs_tol=1e-12)
        assert len(svm.history_["dual_objective"]) == 40

    @STUB
    def test_linearni_svm_na_blobs(self) -> None:
        """Linearni jadro oddeli dva shluky: test >= 0.95, maly pocet SV."""
        data = _blobs_bundle()
        svm = KernelSVM(LinearKernel(), c=1.0, lr=1e-3, n_iters=2000)
        svm.fit(data.x_train, data.y_train)
        assert svm.score(data.x_test, data.y_test) >= 0.95
        assert 0 < len(svm.support_) < len(data.y_train) // 2

    @STUB
    def test_rbf_svm_na_kruznicich(self) -> None:
        """RBF (gamma=1, c=1, lr=1e-3, 3000 iteraci) na kruznicich: test >= 0.95, 30-80 SV."""
        x, y = load_circles(n_samples=200, noise=0.08, factor=0.4, seed=42)
        data = rozdel(x, y, "circles", test_size=0.3, seed=42)
        svm = KernelSVM(RBFKernel(1.0), c=1.0, lr=1e-3, n_iters=3000)
        svm.fit(data.x_train, data.y_train)
        assert svm.score(data.x_test, data.y_test) >= 0.95
        assert 30 <= len(svm.support_) <= 80

    @STUB
    def test_rozhodovaci_funkce_jen_ze_sv(self) -> None:
        """Body s ``lambda = 0`` do ``h(x)`` neprispivaji: SV dava totez co vsechny body."""
        data = _blobs_bundle()
        svm = KernelSVM(LinearKernel(), c=1.0, lr=1e-3, n_iters=500)
        svm.fit(data.x_train, data.y_train)
        a = svm.lambdas_ * svm.y_train_
        h_vse = a @ (data.x_train @ data.x_test.T + 1.0)
        assert torch.allclose(svm.decision_function(data.x_test), h_vse, atol=1e-3)


# --------------------------------------------------------------------------- #
#  Ukladani                                                                   #
# --------------------------------------------------------------------------- #
class TestSaveLoad:
    """``save`` a ``load`` -- slovnik do ``.pt`` a zpet."""

    def _model(self) -> tuple[KernelSVM, DatasetBundle]:
        data = _blobs_bundle()
        model = KernelSVM(LinearKernel(), c=1.0, lr=1e-3, n_iters=300)
        return model.fit(data.x_train, data.y_train), data

    @STUB
    def test_round_trip(self, tmp_path) -> None:
        """Nacteny model ma presne stejne ``h(x)``, jadro, ``c`` a support vektory."""
        model, data = self._model()
        cesta = str(tmp_path / "svm.pt")
        model.save(cesta)
        nacteny = KernelSVM.load(cesta)
        assert isinstance(nacteny, KernelSVM)
        assert type(nacteny.kernel) is LinearKernel
        assert nacteny.c == model.c
        assert torch.equal(nacteny.sv_x_, model.sv_x_)
        assert torch.equal(nacteny.decision_function(data.x_test),
                           model.decision_function(data.x_test))

    @STUB
    def test_rbf_parametry_jadra(self, tmp_path) -> None:
        """U RBF se ulozi a obnovi i ``gamma``."""
        data = _blobs_bundle()
        model = KernelSVM(RBFKernel(0.7), c=1.0, lr=1e-3, n_iters=100)
        model.fit(data.x_train, data.y_train)
        cesta = str(tmp_path / "rbf.pt")
        model.save(cesta)
        nacteny = KernelSVM.load(cesta)
        assert type(nacteny.kernel) is RBFKernel and nacteny.kernel.gamma == 0.7
        assert torch.allclose(nacteny.decision_function(data.x_test),
                              model.decision_function(data.x_test), atol=1e-12)

    @STUB
    def test_soubor_je_slovnik(self, tmp_path) -> None:
        """Soubor je obycejny slovnik (``weights_only=True``) s predepsanymi klici."""
        model, _ = self._model()
        cesta = str(tmp_path / "svm.pt")
        model.save(cesta)
        state = torch.load(cesta, weights_only=True)
        assert isinstance(state, dict)
        assert {"kernel_name", "kernel_params", "c", "sv_x", "sv_y", "sv_lambdas"} <= set(state)
        assert state["kernel_name"] == "linear"

    @STUB
    def test_neexistujici_soubor(self, tmp_path) -> None:
        """Nacteni neexistujiciho souboru je chyba."""
        with pytest.raises((AssertionError, FileNotFoundError)):
            KernelSVM.load(str(tmp_path / "neni.pt"))


# --------------------------------------------------------------------------- #
#  sweep_c                                                                    #
# --------------------------------------------------------------------------- #
class TestSweepC:
    """``sweep_c`` -- vliv regularizace."""

    @STUB
    def test_radky_klice_a_hodnoty(self) -> None:
        """Jeden radek na ``c`` v poradi, klice a typy; sirka pasma se s ``c`` zuzuje."""
        data = _blobs_bundle()
        make = tovarna_modelu(LinearKernel(), _training(lr=1e-3, n_iters=500))
        rows = sweep_c([0.001, 0.01, 1.0], data.x_train, data.y_train,
                       data.x_test, data.y_test, make)
        assert [r["c"] for r in rows] == [0.001, 0.01, 1.0]
        for r in rows:
            assert {"c", "margin", "n_sv", "acc_train", "acc_test"} <= set(r)
            assert isinstance(r["n_sv"], int) and r["margin"] > 0
            assert 0.0 <= r["acc_test"] <= 1.0
        margins = [r["margin"] for r in rows]
        assert margins[0] >= margins[1] >= margins[2]
        assert rows[0]["n_sv"] >= rows[-1]["n_sv"]

    @STUB
    def test_sirka_pasma_bez_jednicky(self) -> None:
        """``margin = 2 / sqrt(a K a)`` s jadrem BEZ ``+1``."""
        data = _blobs_bundle()
        make = tovarna_modelu(LinearKernel(), _training(lr=1e-3, n_iters=300))
        row = sweep_c([1.0], data.x_train, data.y_train, data.x_test, data.y_test, make)[0]
        model = make(1.0).fit(data.x_train, data.y_train)
        a = (model.sv_lambdas_ * model.sv_y_).numpy()
        xs = model.sv_x_.numpy()
        assert math.isclose(row["margin"], 2.0 / math.sqrt(a @ (xs @ xs.T) @ a), rel_tol=1e-9)


def _training(lr: float, n_iters: int):
    """``TrainingConfig`` primo (bez config.yaml)."""
    from dataio import TrainingConfig
    return TrainingConfig(c=1.0, lr=lr, n_iters=n_iters, sv_threshold=1e-5)


# --------------------------------------------------------------------------- #
#  Data a konfigurace (predvyplneno)                                          #
# --------------------------------------------------------------------------- #
class TestDataAKonfigurace:
    """Loadery, rozdeleni se standardizaci a ``config.yaml``."""

    @pytest.mark.parametrize("loader", [load_blobs, load_circles, load_moons])
    def test_2d_loadery(self, loader) -> None:
        """2D hracky: tvar ``(n, 2)``, ``float64``, tridy ``-1`` / ``+1`` vyvazene."""
        x, y = loader(n_samples=100, seed=1)
        assert x.shape == (100, 2) and y.shape == (100,)
        assert x.dtype == torch.float64 and y.dtype == torch.float64
        assert set(y.tolist()) == {-1.0, 1.0}
        assert int((y > 0).sum()) == 50

    def test_breast_cancer(self) -> None:
        """569 x 30, maligni (+1) = 212."""
        x, y = load_breast_cancer()
        assert x.shape == (569, 30)
        assert int((y > 0).sum()) == 212 and int((y < 0).sum()) == 357

    def test_iris_binary(self) -> None:
        """Dve tridy po 50, 4 priznaky."""
        x, y = load_iris_binary()
        assert x.shape == (100, 4) and int((y > 0).sum()) == 50

    def test_split_standardizace_z_treninku(self) -> None:
        """Trenink ma prumer 0 a odchylku 1; stratifikace zachova pomer trid."""
        x, y = load_breast_cancer()
        x_train, y_train, x_test, y_test = train_test_split_std(x, y, test_size=0.3, seed=0)
        assert len(y_train) + len(y_test) == 569
        assert torch.allclose(x_train.mean(0), torch.zeros(30, dtype=torch.float64), atol=1e-10)
        assert torch.allclose(x_train.std(0, unbiased=False), torch.ones(30, dtype=torch.float64))
        podil_train = float((y_train > 0).double().mean())
        podil_test = float((y_test > 0).double().mean())
        assert abs(podil_train - podil_test) < 0.02
        assert not torch.allclose(x_test.mean(0), torch.zeros(30, dtype=torch.float64), atol=1e-10)

    def test_split_neplatny_test_size(self) -> None:
        """``test_size`` mimo ``(0, 1)`` je chyba."""
        x, y = load_blobs()
        with pytest.raises(ValueError):
            train_test_split_std(x, y, test_size=1.0)

    def test_nacteni_configu(self) -> None:
        """Vychozi ``config.yaml`` se nacte a ma ocekavane sekce a hodnoty."""
        cfg = load_config(CONFIG_YAML)
        assert isinstance(cfg, ExperimentConfig)
        assert cfg.kernels.blobs.name == "linear"
        assert cfg.kernels.circles.name == "poly" and cfg.kernels.circles.r == 2
        assert cfg.kernels.moons.name == "rbf"
        assert cfg.training.c == 1.0 and cfg.training.lr is None      # lr: auto
        assert cfg.training.n_iters == 3000
        assert cfg.sweep.c_values == [0.01, 0.1, 1.0, 10.0]
        assert cfg.paths.models_dir == "models" and cfg.paths.logs_dir == "logs"

    @pytest.mark.parametrize("cesta, hodnota, klic", [
        (("training", "c"), 0.0, "training.c"),
        (("training", "lr"), -1.0, "training.lr"),
        (("training", "n_iters"), 0, "training.n_iters"),
        (("data", "test_size"), 1.0, "data.test_size"),
        (("data", "circles_factor"), 1.5, "data.circles_factor"),
        (("kernels", "moons", "gamma"), 0.0, "kernels.moons.gamma"),
        (("kernels", "circles", "r"), 0, "kernels.circles.r"),
        (("kernels", "blobs", "name"), "sigmoid", "kernels.blobs.name"),
        (("diagnostics", "load_tolerance"), 0.0, "diagnostics.load_tolerance"),
        (("seed",), -1, "seed"),
    ])
    def test_validace_jmenuje_klic(self, cesta, hodnota, klic) -> None:
        """Spatna hodnota vyhodi ``ValueError`` se jmenem klice vcetne sekce."""
        cfg = copy.deepcopy(load_config(CONFIG_YAML))
        cil = cfg
        for jmeno in cesta[:-1]:
            cil = getattr(cil, jmeno)
        setattr(cil, cesta[-1], hodnota)
        with pytest.raises(ValueError, match=klic.replace(".", r"\.")):
            validate_config(cfg)

    def test_prazdny_sweep(self) -> None:
        """Prazdny seznam ``c`` je chyba."""
        cfg = copy.deepcopy(load_config(CONFIG_YAML))
        cfg.sweep.c_values = []
        with pytest.raises(ValueError, match="sweep.c_values"):
            validate_config(cfg)

    def test_chybejici_klic(self, tmp_path) -> None:
        """Chybejici klic v YAML vyhodi ``ValueError`` s jeho jmenem."""
        text = open(CONFIG_YAML, encoding="utf-8").read().replace("  n_iters: 3000", "")
        cesta = tmp_path / "config.yaml"
        cesta.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError, match="n_iters"):
            load_config(str(cesta))

    @pytest.mark.parametrize("hodnota, ocekavano", [("auto", None), ("2.0e-3", 2e-3)])
    def test_lr_auto_nebo_cislo(self, tmp_path, hodnota, ocekavano) -> None:
        """``training.lr`` je ``auto`` (-> ``None``) nebo kladne cislo; jinak ``ValueError``."""
        puvodni = open(CONFIG_YAML, encoding="utf-8").read()
        radek = next(r for r in puvodni.splitlines() if r.strip().startswith("lr:"))
        cesta = tmp_path / "config.yaml"
        cesta.write_text(puvodni.replace(radek, f"  lr: {hodnota}"), encoding="utf-8")
        assert load_config(str(cesta)).training.lr == ocekavano
        cesta.write_text(puvodni.replace(radek, "  lr: rychle"), encoding="utf-8")
        with pytest.raises(ValueError, match="training.lr"):
            load_config(str(cesta))

    def test_konfigurace_je_dataclass(self) -> None:
        """Konfigurace je strom dataclass (pristup pres atributy)."""
        cfg = load_config(CONFIG_YAML)
        assert dataclasses.is_dataclass(cfg) and dataclasses.is_dataclass(cfg.kernels.moons)


# --------------------------------------------------------------------------- #
#  utils/ a grafy (predvyplneno, bez studentskeho kodu)                       #
# --------------------------------------------------------------------------- #
class TestPipelineUtils:
    """Predvyplneny balicek ``utils/`` a vykreslovani."""

    def test_divergence(self) -> None:
        """Nekonecne nebo klesajici ``L`` je ``UceniDivergovalo``."""
        diag = load_config(CONFIG_YAML).diagnostics
        with pytest.raises(UceniDivergovalo):
            kontrola_uceni({"dual_objective": [0.1, 5.0, float("nan")]}, diag)
        with pytest.raises(UceniDivergovalo):
            kontrola_uceni({"dual_objective": [0.1, 0.5, -30.0]}, diag)

    def test_varovani_konvergence_z_configu(self, capsys) -> None:
        """Prah varovani se cte z ``diagnostics.convergence_rtol``."""
        diag = load_config(CONFIG_YAML).diagnostics
        hodnoty = {"dual_objective": [float(i) for i in range(1, 101)]}   # +10 % na konci
        kontrola_uceni(hodnoty, dataclasses.replace(diag, convergence_rtol=0.5))
        assert "[POZOR]" not in capsys.readouterr().out
        kontrola_uceni(hodnoty, dataclasses.replace(diag, convergence_rtol=0.05))
        assert "[POZOR]" in capsys.readouterr().out

    def test_tovarna_modelu(self) -> None:
        """Tovarna vytvori nenatrenovany ``KernelSVM`` s danym ``c`` a jadrem (DI)."""
        kernel = LinearKernel()
        model = tovarna_modelu(kernel, _training(lr=0.01, n_iters=7))(2.5)
        assert isinstance(model, KernelSVM)
        assert model.kernel is kernel and model.c == 2.5 and model.n_iters == 7
        assert model.lambdas_ is None

    def test_rozdel_a_radky_sweepu(self) -> None:
        """``rozdel`` vraci ``DatasetBundle``; ``radky_sweepu`` prevede slovniky."""
        x, y = load_moons(n_samples=50, seed=0)
        data = rozdel(x, y, "moons", test_size=0.2, seed=0)
        assert data.name == "moons" and data.x_train.shape == (40, 2)
        radky = radky_sweepu([{"c": 1, "margin": 0.5, "n_sv": 3, "acc_train": 1,
                               "acc_test": 0.9}])
        assert radky == [SweepRow(1.0, 0.5, 3, 1.0, 0.9)]
        with pytest.raises(KeyError):
            radky_sweepu([{"c": 1.0}])

    def test_perceptron(self) -> None:
        """Referencni perceptron oddeli shluky a vraci rozhodovaci funkci ``(m,)``."""
        data = _blobs_bundle()
        fn, acc = nauc_perceptron(data, seed=0)
        assert acc >= 0.9
        assert fn(data.x_test).shape == (len(data.y_test),)

    def test_metriky_proti_sklearn(self) -> None:
        """Kontingencni tabulka a metriky (tridy -1 / +1) souhlasi se ``sklearn.metrics``."""
        g = torch.Generator().manual_seed(3)
        y_true = torch.where(torch.rand(200, generator=g) < 0.4, 1.0, -1.0)
        y_pred = torch.where(torch.rand(200, generator=g) < 0.5, y_true, -y_true)
        yt, yp = y_true.numpy(), y_pred.numpy()
        cm = cm_utils(y_true, y_pred)
        assert np.array_equal(cm, skm.confusion_matrix(yt, yp, labels=[-1, 1]))
        m = vyhodnot_klasifikaci(y_true, y_pred)
        assert math.isclose(m.accuracy, skm.accuracy_score(yt, yp))
        assert math.isclose(m.precision, skm.precision_score(yt, yp, pos_label=1))
        assert math.isclose(m.sensitivity, skm.recall_score(yt, yp, pos_label=1))
        assert math.isclose(m.specificity, skm.recall_score(yt, yp, pos_label=-1))
        assert math.isclose(m.f1, skm.f1_score(yt, yp, pos_label=1))

    def test_metriky_hranicni_pripady(self, capsys) -> None:
        """Bez pozitivnich predikci je precision 0; spatne tridy jsou chyba; vypis ma tabulku."""
        y_true = torch.tensor([1.0, -1.0, 1.0])
        m = vyhodnot_klasifikaci(y_true, torch.tensor([-1.0, -1.0, -1.0]))
        assert m.precision == 0.0 and m.f1 == 0.0 and m.specificity == 1.0
        assert m.cm.tolist() == [[1, 0], [2, 0]]
        with pytest.raises(ValueError):
            cm_utils(torch.tensor([0.0, 1.0]), torch.tensor([1.0, 1.0]))
        with pytest.raises(ValueError):
            cm_utils(torch.tensor([1.0, -1.0]), torch.tensor([1.0]))
        vypis_metriky(m, pozitivni="maligni", negativni="benigni")
        out = capsys.readouterr().out
        assert "TP 0" in out and "FN 2" in out and "propasnuto" in out

    def test_hlasky(self, capsys) -> None:
        """Hlasky maji pevne znacky."""
        faze_neni_hotova(NotImplementedError("Úkol: x"))
        chyba_dat(ValueError("spatna data"))
        preskoceno("duvod")
        out = capsys.readouterr().out
        assert "[NENI HOTOVO] Úkol: x" in out
        assert "[CHYBA DAT] spatna data" in out and "[PRESKOCENO] duvod" in out

    def test_grafy(self, tmp_path) -> None:
        """Vsechny vykreslovaci funkce ulozi PNG (s linearnim vzorem jadra)."""
        x, y = _toy()
        fn = lambda z: z @ torch.tensor([1.0, 1.0], dtype=torch.float64)  # noqa: E731
        cesty = [str(tmp_path / f"{i}.png") for i in range(5)]
        plot_decision_boundary(fn, x, y, cesty[0], sv_x=x[:2])
        plot_boundary_panels([("a", fn, None), ("b", None, x[:1])], x, y, cesty[1])
        plot_gram_heatmap(LinearKernel().gram(x, x), y, cesty[2])
        plot_dual_history({"iteration": [1, 2], "dual_objective": [0.1, 0.2],
                           "n_sv": [4, 3]}, cesty[3])
        plot_c_sweep([{"c": 0.1, "margin": 2.0, "n_sv": 4, "acc_train": 1.0,
                       "acc_test": 1.0},
                      {"c": 1.0, "margin": 1.0, "n_sv": 2, "acc_train": 1.0,
                       "acc_test": 1.0}], cesty[4])
        assert all(os.path.getsize(c) > 0 for c in cesty)
