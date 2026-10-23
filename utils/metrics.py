"""Vyhodnoceni binarni klasifikace z kontingencni tabulky (PREDVYPLNENO).

Jde o orchestraci pipeline ``cviceni_12.py``, kterou student neupravuje --
metriky klasifikace student naprogramoval v Cviceni 06 (``src/metrics.py``),
zde se uz jen pouzivaji. Lezi mimo ``src/`` (latka tohoto cviceni jsou jadra
a SVM) a ``dataio/``; smer zavislosti je ``utils`` -> ``src``, ``dataio``.

Korenovy objekt je 2x2 kontingencni tabulka. Vsechny ostatni metriky
(accuracy, precision, recall/sensitivita, specificita, F1) se pocitaji
POUZE z ni, stejne jako v Cviceni 06.

Rozdil proti Cviceni 06: SVM pracuje s tridami ``-1`` / ``+1``, ne 0 / 1.
Pozitivni trida je ``+1`` (u breast cancer **maligni** nalez), negativni
``-1`` (benigni). Orientace tabulky je stejna (radek = skutecnost,
sloupec = predikce, index 1 = pozitivni)::

                      predikce=-1       predikce=+1
    skutecnost=-1   TN  (cm[0, 0])    FP  (cm[0, 1])
    skutecnost=+1   FN  (cm[1, 0])    TP  (cm[1, 1])

Shoduje se se ``sklearn.metrics.confusion_matrix(y_true, y_pred, labels=[-1, 1])``.

Terminologie: ``recall = sensitivita = Se = TPR = TP / (TP + FN)``,
``specificita = Sp = TNR = TN / (TN + FP)``. V medicinskem kontextu je
klicove nepropasnout maligni nalez (vysoka Se).
"""

from __future__ import annotations

import numpy as np
import torch

from utils.containers import KlasifikacniMetriky


def _as_labels(values: torch.Tensor | np.ndarray, name: str) -> np.ndarray:
    """Pomocna funkce: prevede tridy na 1D pole NumPy a overi hodnoty ``-1`` / ``+1``."""
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    arr = np.asarray(values).ravel()
    if not set(np.unique(arr).tolist()) <= {-1.0, 1.0}:
        raise ValueError(f"{name} musi obsahovat jen tridy -1 a +1")
    return arr


def confusion_matrix(y_true: torch.Tensor | np.ndarray,
                     y_pred: torch.Tensor | np.ndarray) -> np.ndarray:
    """Sestavi 2x2 kontingencni tabulku ``[[TN, FP], [FN, TP]]``.

    Parametry
    ---------
    y_true : torch.Tensor | np.ndarray
        Skutecne tridy ``-1`` / ``+1`` (``+1`` = pozitivni).
    y_pred : torch.Tensor | np.ndarray
        Predikovane tridy ``-1`` / ``+1``, stejne delky jako ``y_true``.

    Navratova hodnota
    -----------------
    np.ndarray
        Celociselna matice tvaru ``(2, 2)`` v orientaci ``[[TN, FP], [FN, TP]]``.

    Vyjimky
    -------
    ``ValueError``:
        Ruzne delky nebo jine tridy nez ``-1`` / ``+1``.
    """
    yt = _as_labels(y_true, "y_true")
    yp = _as_labels(y_pred, "y_pred")
    if yt.shape[0] != yp.shape[0]:
        raise ValueError(f"y_true a y_pred maji ruznou delku ({yt.shape[0]} a {yp.shape[0]})")
    tn = int(np.sum((yt < 0) & (yp < 0)))
    fp = int(np.sum((yt < 0) & (yp > 0)))
    fn = int(np.sum((yt > 0) & (yp < 0)))
    tp = int(np.sum((yt > 0) & (yp > 0)))
    return np.array([[tn, fp], [fn, tp]], dtype=int)


def accuracy(cm: np.ndarray) -> float:
    """Presnost ``(TP + TN) / (TP + TN + FP + FN)``; prazdna tabulka -> ``0.0``.

    Na nevyvazenych datech je zavadejici (viz Cviceni 06).
    """
    tn, fp, fn, tp = (int(v) for v in np.asarray(cm).ravel())
    total = tp + tn + fp + fn
    return (tp + tn) / total if total else 0.0


def precision(cm: np.ndarray) -> float:
    """Preciznost (pozitivni prediktivni hodnota) ``TP / (TP + FP)``; ``0.0`` bez pozitivnich predikci."""
    _, fp, _, tp = (int(v) for v in np.asarray(cm).ravel())
    return tp / (tp + fp) if tp + fp else 0.0


def recall_sensitivity(cm: np.ndarray) -> float:
    """Recall = sensitivita = Se ``TP / (TP + FN)``; ``0.0`` bez pozitivnich vzorku.

    Podil zachycenych pozitivnich (malignich) nalezu -- v klinice klicova metrika.
    """
    _, _, fn, tp = (int(v) for v in np.asarray(cm).ravel())
    return tp / (tp + fn) if tp + fn else 0.0


def specificity(cm: np.ndarray) -> float:
    """Specificita = Sp ``TN / (TN + FP)``; ``0.0`` bez negativnich vzorku."""
    tn, fp, _, _ = (int(v) for v in np.asarray(cm).ravel())
    return tn / (tn + fp) if tn + fp else 0.0


def f1_score(cm: np.ndarray) -> float:
    """F1 skore -- harmonicky prumer preciznosti a recallu; ``0.0``, jsou-li obe nulove."""
    p, r = precision(cm), recall_sensitivity(cm)
    return 2 * p * r / (p + r) if p + r else 0.0


def vyhodnot_klasifikaci(y_true: torch.Tensor | np.ndarray,
                         y_pred: torch.Tensor | np.ndarray) -> KlasifikacniMetriky:
    """Spocita kontingencni tabulku a vsechny metriky najednou.

    Parametry
    ---------
    y_true : torch.Tensor | np.ndarray
        Skutecne tridy ``-1`` / ``+1``.
    y_pred : torch.Tensor | np.ndarray
        Predikovane tridy ``-1`` / ``+1``.

    Navratova hodnota
    -----------------
    KlasifikacniMetriky
        Tabulka a metriky v jedne prepravce.
    """
    cm = confusion_matrix(y_true, y_pred)
    return KlasifikacniMetriky(
        cm=cm,
        accuracy=accuracy(cm),
        precision=precision(cm),
        sensitivity=recall_sensitivity(cm),
        specificity=specificity(cm),
        f1=f1_score(cm),
    )


def vypis_metriky(metriky: KlasifikacniMetriky, pozitivni: str = "pozitivni",
                  negativni: str = "negativni") -> None:
    """Vypise kontingencni tabulku a metriky do konzole.

    Parametry
    ---------
    metriky : KlasifikacniMetriky
        Vystup ``vyhodnot_klasifikaci``.
    pozitivni : str, optional
        Jmeno pozitivni tridy ``+1`` ve vypisu (napr. ``"maligni"``).
    negativni : str, optional
        Jmeno negativni tridy ``-1`` ve vypisu (napr. ``"benigni"``).
    """
    (tn, fp), (fn, tp) = metriky.cm.tolist()
    sirka = max(len(pozitivni), len(negativni)) + 12
    print("  kontingencni tabulka (radek = skutecnost, sloupec = predikce):")
    print(f"  {'':{sirka}s}{'pred. ' + negativni:>18s}{'pred. ' + pozitivni:>18s}")
    print(f"  {'skutecne ' + negativni:{sirka}s}{'TN ' + str(tn):>18s}{'FP ' + str(fp):>18s}")
    print(f"  {'skutecne ' + pozitivni:{sirka}s}{'FN ' + str(fn):>18s}{'TP ' + str(tp):>18s}")
    print(f"  accuracy {metriky.accuracy:.3f} | precision {metriky.precision:.3f} | "
          f"sensitivita (Se) {metriky.sensitivity:.3f} | specificita (Sp) "
          f"{metriky.specificity:.3f} | F1 {metriky.f1:.3f}")
    if fn:
        print(f"  -> {fn} {pozitivni} nalez(u) propasnuto (FN); {fp} falesny(ch) poplach(u) (FP).")
