"""Verejne API balicku ``utils`` pro cviceni 12 (orchestrace pipeline).

Pomocny kod, ktery ridi beh ``cviceni_12.py``: vypisy hlasek, prepravky
na vysledky jednotlivych fazi a spolecne kroky treninku (sestaveni jadra
a modelu z konfigurace, trenink s kontrolou prubehu, referencni perceptron)
a metriky klasifikace z kontingencni tabulky (znate je z Cviceni 06).
Cely balicek je **predvyplneny**, zadny studentsky ukol, zadny
``NotImplementedError``.

Verejne API
-----------
- ``banner`` / ``faze_neni_hotova`` / ``chyba_implementace`` / ``chyba_dat`` /
  ``preskoceno`` -- jednotne hlasky pipeline
- ``DatasetBundle`` / ``TrainedModel`` / ``SweepRow`` / ``KlasifikacniMetriky`` --
  prepravky s vysledky fazi
- ``confusion_matrix`` / ``accuracy`` / ``precision`` / ``recall_sensitivity`` /
  ``specificity`` / ``f1_score`` -- metriky klasifikace (tridy -1 / +1, jako v cv06)
- ``vyhodnot_klasifikaci`` / ``vypis_metriky`` -- vsechny metriky najednou a jejich vypis
- ``UceniDivergovalo`` -- vyjimka pri divergenci treninku
- ``rozdel`` -- stratifikovane rozdeleni + standardizace do ``DatasetBundle``
- ``priprav_logovani`` -- konfigurace loggeru treninku (``logs/svm.log``)
- ``jadro`` / ``tovarna_modelu`` -- jadro a model z konfigurace (pres ``make_kernel``)
- ``kontrola_uceni`` -- divergence a nedokoncena konvergence ``L(lambda)``
- ``train_and_eval`` -- trenink + kontrola + presnost
- ``nauc_perceptron`` -- referencni perceptron pro srovnani
- ``radky_sweepu`` -- vystup ``sweep_c`` jako ``SweepRow``

**Tento soubor neupravujte.**
"""

from __future__ import annotations

from utils.containers import DatasetBundle, KlasifikacniMetriky, SweepRow, TrainedModel
from utils.metrics import (
    accuracy,
    confusion_matrix,
    f1_score,
    precision,
    recall_sensitivity,
    specificity,
    vyhodnot_klasifikaci,
    vypis_metriky,
)
from utils.reporting import (
    banner,
    chyba_dat,
    chyba_implementace,
    faze_neni_hotova,
    preskoceno,
)
from utils.training import (
    UceniDivergovalo,
    jadro,
    kontrola_uceni,
    nauc_perceptron,
    priprav_logovani,
    radky_sweepu,
    rozdel,
    tovarna_modelu,
    train_and_eval,
)

__all__ = [
    "banner",
    "faze_neni_hotova",
    "chyba_implementace",
    "chyba_dat",
    "preskoceno",
    "DatasetBundle",
    "TrainedModel",
    "SweepRow",
    "KlasifikacniMetriky",
    "confusion_matrix",
    "accuracy",
    "precision",
    "recall_sensitivity",
    "specificity",
    "f1_score",
    "vyhodnot_klasifikaci",
    "vypis_metriky",
    "UceniDivergovalo",
    "rozdel",
    "priprav_logovani",
    "jadro",
    "tovarna_modelu",
    "kontrola_uceni",
    "train_and_eval",
    "nauc_perceptron",
    "radky_sweepu",
]
