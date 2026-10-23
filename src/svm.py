"""SVM v dualni formulaci: trenink projektovanym gradientnim vystupem v PyTorch.

Dualni uloha soft-margin SVM (bias foldnuty do jadra, viz nize):

    max_lambda  L(lambda) = sum_i lambda_i
                            - 1/2 sum_ij lambda_i lambda_j y_i y_j k_eff[i, j]
    za podminky 0 <= lambda_i <= c          (box)

Data do ulohy vstupuji JEN pres matici podobnosti ``K`` -- proto staci
vymenit jadro (``LinearKernel`` -> ``RBFKernel``) a cely algoritmus funguje
beze zmeny v jinem prostoru. To je kernelovy trik.

**Bias foldnuty do jadra.** Misto ``h(x) = sum lambda_i y_i K(x_i, x) + b``
pouzivame ``k_eff = K + 1``. Konstanta 1 je skalarni soucin pridane
konstantni slozky priznaku, takze bias se uci spolu s vahami:
``h(x) = sum lambda_i y_i (K(x_i, x) + 1)``, cili ``b = sum lambda_i y_i``.
Diky tomu v dualu odpada rovnostni podminka ``sum lambda_i y_i = 0``
a zustava jen box ``0 <= lambda <= c`` (projekce = obyc. orezani).

Trenink: ``lambda = 0``, pak opakovane ``lambda <- clip(lambda + lr * dL/dlambda, 0, c)``.
Gradient pocita **autograd** (navaznost na cv11), ne rucni vzorec.

Logovani (predvyplneno): pojmenovany logger s ``propagate = False``,
konzole (INFO -- shrnuti kazdeho treninku) a rotujici soubor ``logs/svm.log``
(DEBUG -- prubeh ``L(lambda)`` a poctu support vektoru). Do logu jen skalary.
"""

from __future__ import annotations

import logging
import math
import os
import sys
from logging.handlers import RotatingFileHandler
from typing import Callable, Sequence

import torch
from torch import Tensor

from src.kernels import Kernel, make_kernel

LOG_FILE_NAME = "svm.log"
_LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)-7s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class _ConsoleHandler(logging.StreamHandler):
    """Konzolovy handler, ktery vzdy pise do aktualniho ``sys.stdout`` (PREDVYPLNENO).

    Beznemu ``StreamHandler`` se proud preda jednou pri vytvoreni; pokud jej
    nekdo mezitim vymeni (napr. pytest pri zachytavani vystupu), handler by
    psal do uzavreneho proudu. Tato varianta si proud zjisti pri kazdem zapisu.
    """

    def emit(self, record: logging.LogRecord) -> None:
        """Nastavi aktualni ``sys.stdout`` a zaznam zapise (PREDVYPLNENO)."""
        self.stream = sys.stdout
        super().emit(record)


def get_logger(log_dir: str | None = "logs") -> logging.Logger:
    """Vrati logger modulu, pri prvnim volani ho nakonfiguruje (PREDVYPLNENO).

    Konfigurace probehne **jen jednou** za beh programu (kontrola, zda uz
    logger ma handlery) -- opakovane volani nezpusobi zdvojeny vystup.

    Parametry
    ---------
    log_dir : str | None, optional
        Adresar pro rotujici soubor logu; ``None`` znamena jen konzoli.

    Navratova hodnota
    -----------------
    logging.Logger
        Logger tohoto modulu s urovni DEBUG a ``propagate = False``.
    """
    logger = logging.getLogger(__name__)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # zadne zdvojeni pres korenovy logger

    console = _ConsoleHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("  [%(levelname)s] %(message)s"))
    logger.addHandler(console)

    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = RotatingFileHandler(
            os.path.join(log_dir, LOG_FILE_NAME),
            maxBytes=1_000_000, backupCount=3, encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        logger.addHandler(file_handler)

    return logger


class KernelSVM:
    """Jadrove SVM trenovane v dualu (MIX: smycka predvyplnena, jadro je UKOL).

    Jadro se predava hotove do konstruktoru (dependency injection) -- SVM si
    ho nikdy nevytvari samo, a proto funguje s jakoukoli tridou ``Kernel``.

    Atributy (po ``fit``)
    ---------------------
    lambdas_ : Tensor
        Dualni promenne vsech trenovacich bodu, tvar ``(n,)``.
    x_train_, y_train_ : Tensor
        Trenovaci data.
    support_ : Tensor
        Indexy support vektoru (``lambda_i > sv_threshold``), typ ``int64``.
    sv_x_, sv_y_, sv_lambdas_ : Tensor
        Support vektory, jejich tridy a ``lambda`` -- to je cely model.
    bias_ : float
        Ekvivalentni prah ``b = sum_SV lambda_i y_i`` (dusledek foldnuti +1).
    step_ : float
        Skutecne pouzity krok gradientniho vystupu (``lr``, nebo ``1 / mu_max``
        pri automatickem kroku).
    history_ : dict[str, list[float]]
        Prubeh treninku ve stovce rovnomerne rozlozenych iteraci:
        ``iteration``, ``dual_objective``, ``n_sv``.
    """

    def __init__(
        self,
        kernel: Kernel,
        c: float,
        lr: float | None = None,
        n_iters: int = 3000,
        sv_threshold: float = 1e-5,
    ) -> None:
        """Ulozi jadro a hyperparametry (PREDVYPLNENO).

        Parametry
        ---------
        kernel : Kernel
            Hotova instance jadra (dependency injection).
        c : float
            Regularizace = horni mez boxu ``0 <= lambda <= c``, ``c > 0``.
        lr : float | None, optional
            Krok gradientniho vystupu, ``lr > 0``. ``None`` (vychozi) = automaticky
            krok ``1 / mu_max``, kde ``mu_max`` je nejvetsi vlastni cislo matice
            ``q[i, j] = y_i y_j k_eff[i, j]``; ten je vzdy stabilni (mez je ``2 / mu_max``).
        n_iters : int, optional
            Pocet iteraci, ``>= 1``.
        sv_threshold : float, optional
            Prah, nad kterym je ``lambda_i`` povazovano za nenulove (support vektor).
        """
        if not isinstance(kernel, Kernel):
            raise TypeError(f"kernel musi byt instance tridy Kernel, zadano: {kernel!r}")
        if not (math.isfinite(c) and c > 0):
            raise ValueError(f"c musi byt kladne konecne cislo, zadano: {c}")
        if lr is not None and not (math.isfinite(lr) and lr > 0):
            raise ValueError(f"lr musi byt kladne konecne cislo nebo None (auto), zadano: {lr}")
        if isinstance(n_iters, bool) or not isinstance(n_iters, int) or n_iters < 1:
            raise ValueError(f"n_iters musi byt cele cislo >= 1, zadano: {n_iters!r}")
        if not (math.isfinite(sv_threshold) and sv_threshold >= 0):
            raise ValueError(f"sv_threshold musi byt konecne cislo >= 0, zadano: {sv_threshold}")
        self.kernel = kernel
        self.c = float(c)
        self.lr = None if lr is None else float(lr)
        self.step_: float | None = self.lr      # pri lr=None se urci v ``fit``
        self.n_iters = n_iters
        self.sv_threshold = float(sv_threshold)
        self.logger = logging.getLogger(__name__)

        self.lambdas_: Tensor | None = None
        self.x_train_: Tensor | None = None
        self.y_train_: Tensor | None = None
        self.support_: Tensor | None = None
        self.sv_x_: Tensor | None = None
        self.sv_y_: Tensor | None = None
        self.sv_lambdas_: Tensor | None = None
        self.bias_: float | None = None
        self.history_: dict[str, list[float]] | None = None

    # ------------------------------------------------------------------ #
    #  Bias foldnuty do jadra                                            #
    # ------------------------------------------------------------------ #
    def _effective_gram(self, x: Tensor, z: Tensor) -> Tensor:
        """Matice ``k_eff = K + 1`` -- bias foldnuty do jadra (PREDVYPLNENO).

        Bias foldnuty do kernelu (+1) => v dualu odpada podminka
        ``sum(lambda * y) = 0``, zustava jen box ``0 <= lambda <= c``.

        Parametry
        ---------
        x : Tensor
            Tvar ``(n, d)``.
        z : Tensor
            Tvar ``(m, d)``.

        Navratova hodnota
        -----------------
        Tensor
            ``self.kernel.gram(x, z) + 1.0``, tvar ``(n, m)``.
        """
        return self.kernel.gram(x, z) + 1.0

    # ------------------------------------------------------------------ #
    #  Jadro cviceni: dual a krok gradientniho vystupu                   #
    # ------------------------------------------------------------------ #
    def dual_objective(self, lam: Tensor, k_eff: Tensor, y: Tensor) -> Tensor:
        """Dualni ucelova funkce ``L(lambda)`` (UKOL).

            L(lambda) = sum_i lambda_i - 1/2 sum_ij lambda_i lambda_j y_i y_j k_eff[i, j]

        Prepis vzorce do maticoveho tvaru:

        1. ``q = outer(y, y) * k_eff`` -- matice ``q[i, j] = y_i y_j k_eff[i, j]``
           (``torch.outer``, pak nasobeni PO PRVCICH).
        2. Dvojita suma ``sum_ij lambda_i lambda_j q[i, j]`` je kvadraticka forma
           ``lam @ q @ lam``.
        3. Vratte ``lam.sum() - 0.5 * kvadraticka_forma``.

        Vysledek nechte jako 0-rozmerny TENZOR (ne ``float``, ne ``.item()``,
        ne ``.detach()``) -- autograd z nej v ``_ascent_step`` spocita gradient.

        Parametry
        ---------
        lam : Tensor
            Dualni promenne, tvar ``(n,)``.
        k_eff : Tensor
            Matice ``K + 1`` trenovacich bodu, tvar ``(n, n)``.
        y : Tensor
            Tridy ``-1`` / ``+1``, tvar ``(n,)``.

        Navratova hodnota
        -----------------
        Tensor
            Skalarni (0-rozmerny) tenzor s hodnotou ``L(lambda)``.
        """
        # assert  Ověřte, že lam a y mají tvar (n,) a k_eff tvar (n, n)
        raise NotImplementedError(
            "Úkol: doplnte dual_objective -- prepiste dualni ucelovou funkci L(lambda) "
            "do maticoveho tvaru (postup v docstringu) a vratte ji jako 0-rozmerny tenzor."
        )

    def _ascent_step(self, lam: Tensor, k_eff: Tensor, y: Tensor) -> Tensor:
        """Jeden krok projektovaneho gradientniho VYSTUPU (UKOL, jadro cviceni).

        Dual MAXIMALIZUJEME, proto jdeme PO smeru gradientu (``+``), ne proti
        nemu jako u ztraty v cv11. V tomto poradi:

        1. Vytvorte z ``lam`` list vypocetniho grafu, ktery sleduje gradient:
           ``lam = lam.detach().clone().requires_grad_(True)``.
        2. Spoctete ``dual = self.dual_objective(lam, k_eff, y)``.
        3. Gradient ``dL/dlambda`` ziskejte autogradem: ``dual.backward()``
           (gradient se objevi v ``lam.grad``). Rucne ho NEpocitejte.
        4. V bloku ``with torch.no_grad():`` udelejte krok vystupu
           ``lam + self.step_ * lam.grad`` (``step_`` je krok, ktery pred
           smyckou pripravil ``fit``)
        5. ... a vysledek promitnete zpet do boxu ``0 <= lambda <= c``:
           ``.clamp(0.0, self.c)``.
        6. Vratte nove ``lambda`` jako tenzor BEZ vazby na graf (``.detach()``).

        Parametry
        ---------
        lam : Tensor
            Aktualni ``lambda``, tvar ``(n,)``.
        k_eff : Tensor
            Matice ``K + 1``, tvar ``(n, n)``.
        y : Tensor
            Tridy ``-1`` / ``+1``, tvar ``(n,)``.

        Navratova hodnota
        -----------------
        Tensor
            Nove ``lambda`` v boxu ``[0, c]``, tvar ``(n,)``, ``requires_grad = False``.
        """
        # assert  Ověřte, že gradient lam.grad má stejný tvar jako lam
        raise NotImplementedError(
            "Úkol: doplnte _ascent_step -- jeden krok projektovaneho gradientniho vystupu: "
            "gradient dualu autogradem, krok PO smeru gradientu a projekce do boxu [0, c] "
            "(ocislovane kroky v docstringu)."
        )

    # ------------------------------------------------------------------ #
    #  Treninkova smycka (predvyplneno)                                  #
    # ------------------------------------------------------------------ #
    def fit(self, x: Tensor, y: Tensor) -> "KernelSVM":
        """Natrenuje SVM v dualu a ulozi support vektory (PREDVYPLNENO).

        Smycka jako v cv11, jen misto minimalizace ztraty MAXIMALIZUJE dual:

        1. ``k_eff = K(x, x) + 1`` se spocita JEDNOU (data se behem treninku nemeni).
        2. Krok ``step_``: zadane ``lr``, nebo pri ``lr=None`` automaticky
           ``1 / mu_max`` (nejvetsi vlastni cislo matice ``q = outer(y, y) * k_eff``).
        3. ``lambda = 0`` (zadny bod zatim neni support vektor).
        4. ``n_iters`` krat ``lambda = _ascent_step(lambda, k_eff, y)``; kontrola
           boxu po kazde iteraci, ``L(lambda)`` a pocet support vektoru se do
           historie zapisuji ve stovce rovnomerne rozlozenych iteraci (vypocet
           ``L`` v kazde iteraci by trenink zbytecne zdvojnasobil).
        5. Ulozi se ``lambdas_`` a support vektory (``support_vector_indices``).

        Parametry
        ---------
        x : Tensor
            Trenovaci body, tvar ``(n, d)``.
        y : Tensor
            Tridy ``-1`` / ``+1``, tvar ``(n,)``.

        Navratova hodnota
        -----------------
        KernelSVM
            ``self`` (umoznuje ``model = KernelSVM(...).fit(x, y)``).

        Vyjimky
        -------
        ``ValueError``:
            Spatne tvary nebo tridy, ``_ascent_step`` vratil spatny tvar nebo
            ``lambda`` mimo box ``[0, c]`` (chybi projekce).
        ``TypeError``:
            ``_ascent_step`` nevratil tenzor.
        """
        x = torch.as_tensor(x, dtype=torch.float64)
        y = torch.as_tensor(y, dtype=torch.float64)
        if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.shape[0]:
            raise ValueError(
                f"ocekavano x tvaru (n, d) a y tvaru (n,), zadano {tuple(x.shape)} "
                f"a {tuple(y.shape)}"
            )
        if not bool(torch.all((y == 1.0) | (y == -1.0))):
            raise ValueError("tridy y musi byt -1 nebo +1")

        n = x.shape[0]
        k_eff = self._effective_gram(x, x)
        if self.lr is None:
            mu_max = float(torch.linalg.eigvalsh(torch.outer(y, y) * k_eff).max())
            self.step_ = 1.0 / mu_max
        else:
            self.step_ = self.lr
        lam = torch.zeros(n, dtype=torch.float64)
        history: dict[str, list[float]] = {"iteration": [], "dual_objective": [], "n_sv": []}
        record_every = max(1, self.n_iters // 100)
        log_every = max(1, self.n_iters // 10)
        self.logger.debug(
            "Start treninku: n=%d, d=%d, jadro=%r, c=%g, krok=%.3g (%s), n_iters=%d",
            n, x.shape[1], self.kernel, self.c, self.step_,
            "auto 1/mu_max" if self.lr is None else "lr", self.n_iters,
        )

        for it in range(1, self.n_iters + 1):
            lam = self._ascent_step(lam, k_eff, y)
            if not isinstance(lam, Tensor):
                raise TypeError(
                    f"_ascent_step vratil {type(lam).__name__}, ocekavan tenzor lambda tvaru (n,)"
                )
            if lam.shape != (n,):
                raise ValueError(
                    f"_ascent_step vratil tvar {tuple(lam.shape)}, ocekavan ({n},)"
                )
            lam = lam.detach()
            if float(lam.min()) < -1e-12 or float(lam.max()) > self.c + 1e-12:
                raise ValueError(
                    "_ascent_step vratil lambda mimo box [0, c] -- chybi projekce "
                    "clamp(0, c) po kroku gradientniho vystupu"
                )
            if not (it == 1 or it % record_every == 0 or it == self.n_iters):
                continue
            with torch.no_grad():
                value = float(self.dual_objective(lam, k_eff, y))
            n_sv = int((lam > self.sv_threshold).sum())
            history["iteration"].append(it)
            history["dual_objective"].append(value)
            history["n_sv"].append(n_sv)

            if it == 1 or it % log_every == 0:
                self.logger.debug("iterace %5d | L %.6f | SV %d", it, value, n_sv)
            if not math.isfinite(value):
                self.logger.warning(
                    "L(lambda) neni konecne (%s) -- trenink diverguje, koncim. "
                    "Zkuste mensi lr.", value,
                )
                break

        self.lambdas_ = lam
        self.x_train_ = x
        self.y_train_ = y
        self.history_ = history
        self.support_ = self.support_vector_indices()
        self.sv_x_ = x[self.support_]
        self.sv_y_ = y[self.support_]
        self.sv_lambdas_ = lam[self.support_]
        self.bias_ = float((self.sv_lambdas_ * self.sv_y_).sum())
        self.logger.info(
            "Trenink %r, c=%g, krok %.2e: %d iteraci, L(lambda) = %.4f, "
            "support vektoru %d z %d.",
            self.kernel, self.c, self.step_, history["iteration"][-1],
            history["dual_objective"][-1], len(self.support_), n,
        )
        return self

    def support_vector_indices(self) -> Tensor:
        """Indexy support vektoru: body s ``lambda_i > sv_threshold`` (UKOL, lehky).

        V dualu se support vektory odhali samy: body mimo pasmo maji
        ``lambda_i = 0`` a do rozhodovaci funkce vubec nevstupuji. Kvuli
        konecnemu poctu iteraci porovnavame s malym prahem, ne s presnou nulou.

        Pseudokod:

        * over, ze model je natrenovany (existuji ``lambdas_``)
        * pro kazdy trenovaci bod rozhodni, zda jeho ``lambda`` prekracuje ``sv_threshold``
        * vrat indexy bodu, pro ktere to plati, vzestupne serazene

        Navratova hodnota
        -----------------
        Tensor
            Vzestupne indexy support vektoru, typ ``int64``, tvar ``(n_sv,)``.
        """
        # assert  Ověřte, že model je natrénovaný (self.lambdas_ není None)
        raise NotImplementedError(
            "Úkol: doplnte support_vector_indices -- vratte indexy trenovacich bodu, "
            "jejichz lambda prekracuje prah sv_threshold."
        )

    # ------------------------------------------------------------------ #
    #  Vybavovani                                                        #
    # ------------------------------------------------------------------ #
    def decision_function(self, x: Tensor) -> Tensor:
        """Rozhodovaci funkce ``h(x) = sum_{i in SV} lambda_i y_i (K(x_sv_i, x) + 1)`` (UKOL).

        Pouzivaji se JEN support vektory (``self.sv_x_``, ``self.sv_y_``,
        ``self.sv_lambdas_``) -- ostatni body maji ``lambda = 0`` a nic
        nepridavaji.

        Pseudokod (pro vsechny body ``x`` najednou, bez smycky):

        * over, ze model je natrenovany (existuji support vektory)
        * spocti efektivni podobnost (jadro + 1) kazdeho support vektoru s kazdym
          bodem ``x`` -- matice ``(n_sv, m)``
        * vaha support vektoru i je ``lambda_i * y_i`` -- vektor ``(n_sv,)``
        * ``h`` kazdeho bodu je vazeny soucet jeho podobnosti se support vektory
        * vrat ``h`` pro vsechny body, tvar ``(m,)``

        Znamenko ``h(x)`` urcuje tridu, ``|h(x)|`` vzdalenost od hranice
        (``h = +-1`` jsou okraje pasma).

        Parametry
        ---------
        x : Tensor
            Body k vyhodnoceni, tvar ``(m, d)``.

        Navratova hodnota
        -----------------
        Tensor
            Hodnoty ``h(x)``, tvar ``(m,)``.
        """
        # assert  Ověřte, že model je natrénovaný (self.sv_x_ není None)
        raise NotImplementedError(
            "Úkol: doplnte decision_function -- spoctete h(x) pro vsechny body x ze "
            "support vektoru a efektivniho jadra K + 1 (postup v docstringu)."
        )

    def predict(self, x: Tensor) -> Tensor:
        """Predikce tridy ``-1`` / ``+1`` podle znamenka ``h(x)`` (PREDVYPLNENO).

        Bod presne na hranici (``h = 0``) dostane tridu ``+1``.

        Parametry
        ---------
        x : Tensor
            Tvar ``(m, d)``.

        Navratova hodnota
        -----------------
        Tensor
            Tridy, tvar ``(m,)``, typ ``float64``.
        """
        x = torch.as_tensor(x, dtype=torch.float64)
        h = self.decision_function(x)
        return torch.where(h >= 0, 1.0, -1.0).to(torch.float64)

    def score(self, x: Tensor, y: Tensor) -> float:
        """Presnost (accuracy) -- podil spravne klasifikovanych bodu (PREDVYPLNENO).

        Parametry
        ---------
        x : Tensor
            Tvar ``(m, d)``.
        y : Tensor
            Skutecne tridy ``-1`` / ``+1``, tvar ``(m,)``.

        Navratova hodnota
        -----------------
        float
            Presnost v intervalu ``[0, 1]``.
        """
        y = torch.as_tensor(y, dtype=torch.float64)
        return float((self.predict(x) == y).to(torch.float64).mean())

    # ------------------------------------------------------------------ #
    #  Perzistence                                                       #
    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        """Ulozi natrenovany model do souboru ``.pt`` (UKOL).

        SVM je induktivni model: naucene pravidlo (support vektory a jejich
        ``lambda``) se da pouzit na nova data -- ma tedy co ulozit. Ulozte
        ``torch.save`` OBYCEJNY SLOVNIK (ne cely objekt -- pickle objektu je
        cerna skrinka a bezpecnostni riziko), ktery obsahuje vse, co
        potrebuje ``decision_function`` nove nactene instance:

        * ``"kernel_name"``: ``self.kernel.name`` (``"linear"`` / ``"poly"`` / ``"rbf"``),
        * ``"kernel_params"``: parametry jadra jako slovnik (``vars(self.kernel)``,
          napr. ``{"gamma": 1.0}``),
        * ``"c"``, ``"lr"``, ``"n_iters"``, ``"sv_threshold"``: hyperparametry,
        * ``"sv_x"``, ``"sv_y"``, ``"sv_lambdas"``: support vektory, jejich tridy a ``lambda``,
        * ``"bias"``: ``self.bias_`` (informativne; v ``h(x)`` je obsazen pres ``+1``).

        Parametry
        ---------
        path : str
            Cesta k souboru, napr. ``models/svm_breast_cancer.pt``.
        """
        # assert  Ověřte, že model je natrénovaný (self.sv_x_ není None)
        # assert  Ověřte, že path končí příponou .pt
        raise NotImplementedError(
            "Úkol: doplnte save -- ulozte do path obycejny slovnik se vsim, co nactena "
            "instance potrebuje k predikci (seznam klicu v docstringu)."
        )

    @classmethod
    def load(cls, path: str) -> "KernelSVM":
        """Nacte model ulozeny metodou ``save`` a vrati novou instanci (UKOL).

        1. ``state = torch.load(path, weights_only=True)`` -- bezpecne nacteni
           jen tenzoru a zakladnich typu (zadny libovolny kod).
        2. Jadro sestavte tovarni funkci ``make_kernel(state["kernel_name"],
           **state["kernel_params"])``.
        3. ``model = cls(kernel, c, lr, n_iters, sv_threshold)`` z ulozenych hodnot.
        4. Nastavte ``model.sv_x_``, ``model.sv_y_``, ``model.sv_lambdas_``
           a ``model.bias_``; vratte ``model``.

        Nacteny model umi ``decision_function`` a ``predict`` (``lambdas_``
        vsech trenovacich bodu se neukladaji -- k predikci nejsou potreba).

        Parametry
        ---------
        path : str
            Cesta k souboru ``.pt``.

        Navratova hodnota
        -----------------
        KernelSVM
            Model s predikcemi shodnymi s ulozenym.
        """
        # assert  Ověřte, že soubor path existuje
        raise NotImplementedError(
            "Úkol: doplnte load -- bezpecne nactete slovnik ze souboru, sestavte z nej "
            "jadro a novou instanci modelu se support vektory."
        )


def sweep_c(
    c_values: Sequence[float],
    x_train: Tensor,
    y_train: Tensor,
    x_test: Tensor,
    y_test: Tensor,
    make_model: Callable[[float], KernelSVM],
) -> list[dict]:
    """Vliv regularizace ``c``: natrenuje model pro kazde ``c`` a sebere metriky (UKOL, lehky).

    Pro kazde ``c`` z ``c_values`` (v zadanem poradi):

    1. ``model = make_model(c).fit(x_train, y_train)`` -- tovarna dodana
       zvenku vytvori NENATRENOVANY model se spravnym jadrem (DI).
    2. Sirka pasma ``margin = 2 / ||w||``, kde
       ``||w||^2 = sum_ij lambda_i lambda_j y_i y_j K[i, j]`` pres support
       vektory a ``k = model.kernel.gram(sv_x_, sv_x_)`` -- BEZ ``+1``
       (bias do ``||w||`` nepatri). Maticove: ``a @ k @ a`` pro
       ``a = sv_lambdas_ * sv_y_``.
    3. Pocet support vektoru ``len(model.support_)``.
    4. Presnost ``model.score`` na trenovacich i testovacich datech.

    Kazde ``c`` da jeden slovnik s klici ``"c"``, ``"margin"``, ``"n_sv"``,
    ``"acc_train"``, ``"acc_test"`` (hodnoty python ``float``, ``n_sv`` ``int``).

    Parametry
    ---------
    c_values : Sequence[float]
        Testovane hodnoty ``c``.
    x_train, y_train : Tensor
        Trenovaci data.
    x_test, y_test : Tensor
        Testovaci data.
    make_model : Callable[[float], KernelSVM]
        Funkce ``c -> KernelSVM`` (nenatrenovany model).

    Navratova hodnota
    -----------------
    list[dict]
        Jeden radek (slovnik) na kazde ``c``, ve stejnem poradi jako ``c_values``.
    """
    # assert  Ověřte, že c_values není prázdné
    raise NotImplementedError(
        "Úkol: doplnte sweep_c -- pro kazde c natrenujte model a zaznamenejte sirku "
        "pasma, pocet support vektoru a presnost na treninku i testu."
    )
