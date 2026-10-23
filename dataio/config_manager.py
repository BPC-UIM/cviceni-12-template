"""Typovana sprava konfigurace nad ``config.yaml`` pro cviceni 12 (PREDVYPLNENO).

Zasada: **vse nastavitelne je v ``config.yaml``** -- cesty, parametry dat,
jadra jednotlivych fazi, hyperparametry treninku, hodnoty ``c`` a ``gamma``
pro experimenty i prahy kontrol; v kodu pipeline nejsou "magicke" konstanty.

Modul definuje dataclassy odpovidajici sekcim ``config.yaml`` (``paths``,
``data``, ``kernels``, ``training``, ``sweep``, ``diagnostics`` a korenovy
``seed``) a dve funkce: ``load_config`` (naparsuje YAML, sestavi dataclassy,
zvaliduje a vrati) a ``validate_config`` (rozsahove kontroly s ceskymi
hlaskami, ktere jmenuji klic vcetne sekce, napr. ``kernels.moons.gamma``).

K hodnotam se pristupuje pres atributy (napr. ``cfg.kernels.moons.gamma``),
nikdy ne pres klice slovniku -- preklep v atributu odhali editor staticky,
zatimco ``cfg["kernels"]["moons"]["gamma"]`` spadne az za behu.

Jmeno jadra (``linear`` / ``poly`` / ``rbf``) se v pipeline predava tovarni
funkci ``make_kernel`` (vzor Strategy + Factory) -- jadro dane faze tedy
zmenite jen upravou ``config.yaml``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import yaml

KERNEL_NAMES = ("linear", "poly", "rbf")


@dataclass
class PathsConfig:
    """Vystupni slozky (sekce ``paths``), relativne ke koreni repozitare.

    Atributy
    --------
    graphs_dir:
        Slozka pro vystupni grafy (.png).
    models_dir:
        Slozka pro ulozene modely (.pt).
    logs_dir:
        Slozka s logem treninku (``svm.log``).
    """

    graphs_dir: str
    models_dir: str
    logs_dir: str


@dataclass
class DataConfig:
    """Parametry dat (sekce ``data``).

    Atributy
    --------
    n_samples:
        Pocet bodu kazdeho 2D datasetu (blobs, circles, moons), ``>= 20``.
    test_size:
        Podil testovacich bodu (stratifikovane rozdeleni), z ``(0, 1)``.
    blobs_std:
        Smerodatna odchylka shluku v ``blobs``, ``> 0``.
    circles_noise:
        Sum kruznic, ``>= 0``.
    circles_factor:
        Pomer polomeru vnitrni a vnejsi kruznice, z ``(0, 1)``.
    moons_noise:
        Sum pulmesicu, ``>= 0``.
    """

    n_samples: int
    test_size: float
    blobs_std: float
    circles_noise: float
    circles_factor: float
    moons_noise: float


@dataclass
class KernelConfig:
    """Volba jadra jedne faze.

    Atributy
    --------
    name:
        ``"linear"``, ``"poly"`` nebo ``"rbf"`` (jmeno pro ``make_kernel``).
    r:
        Stupen polynomu (pouzije se jen u ``poly``), cele cislo ``>= 1``.
    gamma:
        Sirka RBF jadra (pouzije se jen u ``rbf``), ``> 0``.
    """

    name: str
    r: int
    gamma: float


@dataclass
class KernelsConfig:
    """Jadra jednotlivych fazi pipeline (sekce ``kernels``).

    Atributy
    --------
    blobs:
        Jadro faze A1 (shluky; vychozi ``linear``).
    circles:
        Jadro faze A2 (kruznice; vychozi ``poly``, ``r = 2``).
    moons:
        Jadro faze A3 (pulmesice; vychozi ``rbf``).
    breast_cancer:
        Jadro faze B (realna 30rozmerna data; vybira se experimentem).
    """

    blobs: KernelConfig
    circles: KernelConfig
    moons: KernelConfig
    breast_cancer: KernelConfig


@dataclass
class TrainingConfig:
    """Hyperparametry dualniho treninku (sekce ``training``).

    Atributy
    --------
    c:
        Regularizace = horni mez boxu ``0 <= lambda <= c``, ``> 0``.
    lr:
        Krok projektovaneho gradientniho vystupu, ``> 0``; ``None`` (v YAML
        ``lr: auto``) = automaticky stabilni krok ``1 / mu_max``.
    n_iters:
        Pocet iteraci, ``>= 1``.
    sv_threshold:
        Prah ``lambda``, nad kterym je bod support vektorem, ``>= 0``.
    """

    c: float
    lr: float | None
    n_iters: int
    sv_threshold: float


@dataclass
class SweepConfig:
    """Hodnoty pro experimenty s ``c`` a ``gamma`` (sekce ``sweep``).

    Atributy
    --------
    c_values:
        Hodnoty ``c`` pro ``sweep_c`` na breast cancer (faze B2), kazda ``> 0``.
    grid_c_values:
        Hodnoty ``c`` mrizky ``c x gamma`` na pulmesicich (faze A3).
    grid_gamma_values:
        Hodnoty ``gamma`` mrizky ``c x gamma`` na pulmesicich (faze A3).
    """

    c_values: list[float]
    grid_c_values: list[float]
    grid_gamma_values: list[float]


@dataclass
class DiagnosticsConfig:
    """Prahy kontrol, ktere pipeline vypisuje (sekce ``diagnostics``).

    Atributy
    --------
    trick_tolerance:
        Nejvetsi povoleny rozdil ``gram(x, z)`` a ``phi(x) phi(z)^T``, ``> 0``.
    load_tolerance:
        Nejvetsi povoleny rozdil ``h(x)`` puvodniho a nacteneho modelu, ``> 0``.
    convergence_rtol:
        Relativni narust ``L(lambda)`` za poslednich 10 % iteraci, nad kterym
        pipeline varuje, ze trenink nedokonvergoval (``0.05`` = 5 %), ``> 0``.
    """

    trick_tolerance: float
    load_tolerance: float
    convergence_rtol: float


@dataclass
class ExperimentConfig:
    """Korenova konfigurace experimentu slozena ze vsech sekci.

    Atributy
    --------
    paths, data, kernels, training, sweep, diagnostics:
        Dilci sekce (viz jednotlive dataclassy).
    seed:
        Spolecny seed generovani a rozdeleni dat, ``>= 0``.
    """

    paths: PathsConfig
    data: DataConfig
    kernels: KernelsConfig
    training: TrainingConfig
    sweep: SweepConfig
    diagnostics: DiagnosticsConfig
    seed: int


# --------------------------------------------------------------------------- #
#  Pomocne prevody typu                                                       #
# --------------------------------------------------------------------------- #
def _as_float(name: str, value: Any) -> float:
    """Pomocna funkce: prevede hodnotu na ``float``; bool a necisla vyhodi ``ValueError``.

    Retezec se prijme, jen kdyz je to zapis cisla (YAML cte ``1e-3`` jako retezec).
    """
    if isinstance(value, bool):
        raise ValueError(f"{name} musi byt cislo, zadano: {value!r}")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            pass
    raise ValueError(f"{name} musi byt cislo, zadano: {value!r}")


def _as_int(name: str, value: Any) -> int:
    """Pomocna funkce: vrati ``int``; ``float``, ``bool`` a retezec vyhodi ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} musi byt cele cislo, zadano: {value!r}")
    return value


def _as_lr(name: str, value: Any) -> float | None:
    """Pomocna funkce: krok uceni -- retezec ``auto`` -> ``None``, jinak cislo."""
    if isinstance(value, str) and value.strip().lower() == "auto":
        return None
    try:
        return _as_float(name, value)
    except ValueError:
        raise ValueError(f"{name} musi byt kladne cislo nebo 'auto', zadano: {value!r}") from None


def _as_str(name: str, value: Any) -> str:
    """Pomocna funkce: vrati retezec; cokoli jineho vyhodi ``ValueError``."""
    if not isinstance(value, str):
        raise ValueError(f"{name} musi byt retezec, zadano: {value!r}")
    return value


def _as_float_list(name: str, value: Any) -> list[float]:
    """Pomocna funkce: vrati seznam cisel; jinak ``ValueError``."""
    if not isinstance(value, list):
        raise ValueError(f"{name} musi byt seznam cisel, zadano: {value!r}")
    return [_as_float(f"{name}[{i}]", item) for i, item in enumerate(value)]


def _get(section: dict[str, Any], section_name: str | None, key: str) -> Any:
    """Pomocna funkce: vrati ``section[key]``; chybejici klic -> ``ValueError``."""
    if key not in section:
        where = f"v sekci {section_name}" if section_name else "v korenu souboru"
        raise ValueError(f"{where} chybi klic '{key}'")
    return section[key]


def _get_section(raw: dict[str, Any], name: str, where: str) -> dict[str, Any]:
    """Pomocna funkce: vrati podsekci ``name``; chybi-li nebo neni slovnik -> ``ValueError``."""
    if name not in raw:
        raise ValueError(f"{where}: chybi sekce '{name}'")
    section = raw[name]
    if not isinstance(section, dict):
        raise ValueError(f"{where}: sekce '{name}' musi obsahovat klice, zadano: {section!r}")
    return section


def _make_kernel_config(section: dict[str, Any], name: str) -> KernelConfig:
    """Pomocna funkce: slozi ``KernelConfig`` z podsekce ``kernels.<faze>``."""
    return KernelConfig(
        name=_as_str(f"{name}.name", _get(section, name, "name")),
        r=_as_int(f"{name}.r", _get(section, name, "r")),
        gamma=_as_float(f"{name}.gamma", _get(section, name, "gamma")),
    )


def load_config(filepath: str = "config.yaml") -> ExperimentConfig:
    """Nacte a zvaliduje konfiguraci z YAML souboru.

    Parametry
    ---------
    filepath:
        Cesta k YAML souboru s konfiguraci.

    Navratova hodnota
    -----------------
    ``ExperimentConfig`` se vsemi sekcemi.

    Vyjimky
    -------
    ``FileNotFoundError``:
        Pokud soubor neexistuje.
    ``ValueError``:
        Pokud YAML nelze naparsovat, chybi sekce/klic, klic ma spatny typ,
        nebo hodnota nesplnuje kontroly ve ``validate_config``.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as handle:
            raw: Any = yaml.safe_load(handle)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Konfiguracni soubor '{filepath}' neexistuje (spoustejte skript z korene repozitare)"
        ) from exc
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        line = f" na radku {mark.line + 1}" if mark is not None else ""
        problem = getattr(exc, "problem", None) or str(exc)
        raise ValueError(
            f"{filepath} neni platny YAML{line}: {problem}. "
            "Pozor, v odsazeni nesmi byt tabulatory, jen mezery."
        ) from exc
    except UnicodeDecodeError as exc:
        raise ValueError(f"{filepath} nelze precist jako text UTF-8: {exc.reason}") from exc

    if not isinstance(raw, dict):
        raise ValueError(
            f"{filepath} nema ocekavanou strukturu: koren musi obsahovat sekce "
            f"paths, data, kernels, training, sweep, diagnostics a klic seed (nacteno: {raw!r})"
        )

    paths = _get_section(raw, "paths", filepath)
    data = _get_section(raw, "data", filepath)
    kernels = _get_section(raw, "kernels", filepath)
    training = _get_section(raw, "training", filepath)
    sweep = _get_section(raw, "sweep", filepath)
    diag = _get_section(raw, "diagnostics", filepath)

    cfg = ExperimentConfig(
        paths=PathsConfig(
            **{k: _as_str(f"paths.{k}", _get(paths, "paths", k))
               for k in ("graphs_dir", "models_dir", "logs_dir")}
        ),
        data=DataConfig(
            n_samples=_as_int("data.n_samples", _get(data, "data", "n_samples")),
            **{k: _as_float(f"data.{k}", _get(data, "data", k))
               for k in ("test_size", "blobs_std", "circles_noise", "circles_factor",
                         "moons_noise")}
        ),
        kernels=KernelsConfig(
            **{faze: _make_kernel_config(
                _get_section(kernels, faze, "sekce kernels"), f"kernels.{faze}")
               for faze in ("blobs", "circles", "moons", "breast_cancer")}
        ),
        training=TrainingConfig(
            c=_as_float("training.c", _get(training, "training", "c")),
            lr=_as_lr("training.lr", _get(training, "training", "lr")),
            n_iters=_as_int("training.n_iters", _get(training, "training", "n_iters")),
            sv_threshold=_as_float("training.sv_threshold",
                                   _get(training, "training", "sv_threshold")),
        ),
        sweep=SweepConfig(
            **{k: _as_float_list(f"sweep.{k}", _get(sweep, "sweep", k))
               for k in ("c_values", "grid_c_values", "grid_gamma_values")}
        ),
        diagnostics=DiagnosticsConfig(
            **{k: _as_float(f"diagnostics.{k}", _get(diag, "diagnostics", k))
               for k in ("trick_tolerance", "load_tolerance", "convergence_rtol")}
        ),
        seed=_as_int("seed", _get(raw, None, "seed")),
    )
    validate_config(cfg)
    return cfg


def _check_positive(name: str, value: float) -> None:
    """Pomocna funkce: hodnota musi byt konecne cislo ``> 0``."""
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} musi byt konecne cislo > 0, zadano: {value}")


def _check_non_negative(name: str, value: float) -> None:
    """Pomocna funkce: hodnota musi byt konecne cislo ``>= 0``."""
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} musi byt konecne cislo >= 0, zadano: {value}")


def _check_open_unit(name: str, value: float) -> None:
    """Pomocna funkce: hodnota musi lezet v otevrenem intervalu ``(0, 1)``."""
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} musi lezet v intervalu (0, 1), zadano: {value}")


def _validate_kernel(name: str, kernel: KernelConfig) -> None:
    """Pomocna funkce: jmeno z registru, ``r >= 1``, ``gamma > 0``."""
    if kernel.name not in KERNEL_NAMES:
        raise ValueError(
            f"{name}.name musi byt jedno z {', '.join(KERNEL_NAMES)}, zadano: {kernel.name!r}"
        )
    if isinstance(kernel.r, bool) or not isinstance(kernel.r, int) or kernel.r < 1:
        raise ValueError(f"{name}.r musi byt cele cislo >= 1, zadano: {kernel.r!r}")
    _check_positive(f"{name}.gamma", kernel.gamma)


def _validate_positive_list(name: str, values: list[float]) -> None:
    """Pomocna funkce: neprazdny seznam kladnych konecnych cisel."""
    if len(values) == 0:
        raise ValueError(f"{name} musi byt neprazdny seznam kladnych cisel, zadano: {values}")
    for i, value in enumerate(values):
        _check_positive(f"{name}[{i}]", value)


def validate_config(cfg: ExperimentConfig) -> None:
    """Zkontroluje rozsahy hodnot v konfiguraci.

    Pri poruseni nektere podminky vyhodi ``ValueError`` s ceskou hlaskou,
    ktera jmenuje klic vcetne sekce a zadanou hodnotu. Kontroluje se:

    - ``paths.*`` neprazdne retezce
    - ``data.n_samples >= 20``, ``0 < data.test_size < 1``, ``data.blobs_std > 0``,
      ``data.circles_noise >= 0``, ``0 < data.circles_factor < 1``, ``data.moons_noise >= 0``
    - ``kernels.<faze>.name`` z ``{linear, poly, rbf}``, ``r >= 1``, ``gamma > 0``
    - ``training.c > 0``, ``training.lr > 0`` nebo ``auto``, ``training.n_iters >= 1``,
      ``training.sv_threshold >= 0``
    - ``sweep.*`` neprazdne seznamy kladnych cisel
    - ``diagnostics.*`` kladna konecna cisla
    - ``seed`` cele cislo ``>= 0``

    Navratova hodnota je ``None`` -- funkce pouze validuje.
    """
    for key in ("graphs_dir", "models_dir", "logs_dir"):
        value = getattr(cfg.paths, key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"paths.{key} musi byt neprazdny retezec, zadano: {value!r}")

    data = cfg.data
    if isinstance(data.n_samples, bool) or not isinstance(data.n_samples, int) \
            or data.n_samples < 20:
        raise ValueError(f"data.n_samples musi byt cele cislo >= 20, zadano: {data.n_samples!r}")
    _check_open_unit("data.test_size", data.test_size)
    _check_positive("data.blobs_std", data.blobs_std)
    _check_non_negative("data.circles_noise", data.circles_noise)
    _check_open_unit("data.circles_factor", data.circles_factor)
    _check_non_negative("data.moons_noise", data.moons_noise)

    for faze in ("blobs", "circles", "moons", "breast_cancer"):
        _validate_kernel(f"kernels.{faze}", getattr(cfg.kernels, faze))

    tr = cfg.training
    _check_positive("training.c", tr.c)
    if tr.lr is not None:
        _check_positive("training.lr", tr.lr)
    if isinstance(tr.n_iters, bool) or not isinstance(tr.n_iters, int) or tr.n_iters < 1:
        raise ValueError(f"training.n_iters musi byt cele cislo >= 1, zadano: {tr.n_iters!r}")
    _check_non_negative("training.sv_threshold", tr.sv_threshold)

    for key in ("c_values", "grid_c_values", "grid_gamma_values"):
        _validate_positive_list(f"sweep.{key}", getattr(cfg.sweep, key))

    _check_positive("diagnostics.trick_tolerance", cfg.diagnostics.trick_tolerance)
    _check_positive("diagnostics.load_tolerance", cfg.diagnostics.load_tolerance)
    _check_positive("diagnostics.convergence_rtol", cfg.diagnostics.convergence_rtol)

    if isinstance(cfg.seed, bool) or not isinstance(cfg.seed, int) or cfg.seed < 0:
        raise ValueError(f"seed musi byt cele cislo >= 0, zadano: {cfg.seed!r}")
