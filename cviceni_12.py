# -*- coding: utf-8 -*-

"""
Created on 14. 10. 2026 at 10:00:00

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
    Vstupni bod cviceni 12 (SVM a kernelovy trik, trenink v dualu, PyTorch).
    Pipeline ma dve casti a sest fazi.

    Cast A -- 2D hracky (strukturu je VIDET):
      A1. linearni jadro na dvou shlucich (blobs): hranice, pasmo, support
          vektory, prubeh L(lambda), heatmapa Gramovy matice,
      A2. polynomialni jadro na kruznicich (circles): overeni kernelove triku
          K = phi . phi, trenink a kontrast s linearnim jadrem,
      A3. RBF jadro na pulmesicich (moons): perceptron vs. linearni vs.
          jadrove SVM a mrizka c x gamma.

    Cast B -- breast cancer (30 priznaku, "slepa" realna uloha):
      B1. data a trenink s jadrem z config.yaml, kontingencni tabulka a metriky
          klasifikace (accuracy, precision, Se, Sp, F1 -- utils/metrics.py),
      B2. vliv regularizace c (sweep_c): sirka pasma, pocet SV, presnost,
      B3. ulozeni modelu do models/ a kontrolni nacteni.

    Vse nastavitelne (cesty, data, jadra fazi, hyperparametry treninku,
    hodnoty c a gamma, prahy kontrol) je v config.yaml; tento soubor
    neobsahuje zadne konstanty. Pomocny kod pipeline (hlasky, prepravky,
    sestaveni modelu, trenink s kontrolou, metriky klasifikace) je v balicku
    utils/.

    Cela pipeline je predvyplnena. Studentske ukoly jsou v src/kernels.py
    a src/svm.py -- viz README, Pokyny k vypracovani.

    Repozitar bezi v kazdem stavu. Dokud nejsou ukoly hotove, faze se
    zastavi jen hlaskou [NENI HOTOVO] Úkol: ... a pipeline pokracuje dal;
    faze zavisla na neuspesne fazi se ohlasi jako [PRESKOCENO]. Jakakoli
    jina vyjimka z faze je [CHYBA IMPLEMENTACE], chyba dat [CHYBA DAT],
    divergence treninku [CHYBA UCENI]. Nikdy nezpracovany traceback.
================================================================================
"""

from __future__ import annotations

import os
import pickle
import sys

import torch

# --- Import guard: srozumitelna hlaska misto holeho ImportError ----------------
try:
    from dataio import (
        ExperimentConfig,
        load_blobs,
        load_breast_cancer,
        load_circles,
        load_config,
        load_moons,
        plot_boundary_panels,
        plot_c_sweep,
        plot_decision_boundary,
        plot_dual_history,
        plot_gram_heatmap,
    )
    from src import (
        KernelSVM,
        LinearKernel,
        PolynomialKernel,
        RBFKernel,
        sweep_c,
        verify_kernel_trick,
    )
    from utils import (
        DatasetBundle,
        TrainedModel,
        UceniDivergovalo,
        banner,
        chyba_dat,
        chyba_implementace,
        faze_neni_hotova,
        jadro,
        nauc_perceptron,
        preskoceno,
        priprav_logovani,
        radky_sweepu,
        rozdel,
        tovarna_modelu,
        train_and_eval,
        vyhodnot_klasifikaci,
        vypis_metriky,
    )
except ImportError as exc:  # pragma: no cover - jen ochranna hlaska
    print(f"[CHYBA IMPORTU] Nepodarilo se nacist moduly projektu: {exc}")
    print("Zkontrolujte, ze spoustite skript z korene repozitare a mate "
          "nainstalovane zavislosti (pip install -r requirements.txt).")
    sys.exit(1)


# ============================================================================ #
#  Cast A: 2D hracky                                                           #
# ============================================================================ #
def faze_a1_linearni(cfg: ExperimentConfig) -> None:
    """Faze A1 -- linearni jadro na dvou shlucich: hranice, pasmo, SV, Gramova matice.

    Parametry
    ---------
    cfg : ExperimentConfig
        Konfigurace experimentu.
    """
    banner("Faze A1: Linearni SVM -- dva shluky (blobs)")
    try:
        x, y = load_blobs(cfg.data.n_samples, cfg.data.blobs_std, cfg.seed)
        data = rozdel(x, y, "blobs", cfg.data.test_size, cfg.seed)
    except Exception as exc:
        chyba_dat(exc)
        print("  -> Zkontrolujte sekci data v config.yaml; faze A1 se preskoci.")
        return
    graphs = cfg.paths.graphs_dir
    print(f"  data: trenink {tuple(data.x_train.shape)}, test {tuple(data.x_test.shape)}")
    try:
        kernel = jadro(cfg.kernels.blobs)
        krok = "auto (1/mu_max)" if cfg.training.lr is None else cfg.training.lr
        print(f"  jadro: {kernel!r}; c = {cfg.training.c}, krok {krok}, "
              f"iteraci {cfg.training.n_iters}")
        gram_png = os.path.join(graphs, "a1_gram_blobs.png")
        plot_gram_heatmap(kernel.gram(data.x_train, data.x_train), data.y_train, gram_png,
                          title=f"Gramova matice {kernel!r} (blobs, trenink)")
        print(f"  graf ulozen: {gram_png}")

        vysledek = train_and_eval(tovarna_modelu(kernel, cfg.training), data,
                                  cfg.training.c, cfg.diagnostics)
        model = vysledek.model
        print(f"  presnost: trenink {vysledek.acc_tr:.3f}, test {vysledek.acc_te:.3f}")
        print(f"  support vektoru: {vysledek.n_sv} z {len(data.y_train)}; "
              f"bias b = sum lambda_i y_i = {model.bias_:.4f}")
        hranice_png = os.path.join(graphs, "a1_hranice_blobs.png")
        prubeh_png = os.path.join(graphs, "a1_prubeh_dualu.png")
        plot_decision_boundary(model.decision_function, data.x_train, data.y_train, hranice_png,
                               title=f"A1: {kernel!r}, c = {cfg.training.c}", sv_x=model.sv_x_)
        plot_dual_history(model.history_, prubeh_png, title="A1: prubeh treninku v dualu")
        print(f"  grafy ulozeny: {hranice_png}, {prubeh_png}")
    except NotImplementedError as exc:
        faze_neni_hotova(exc)
    except UceniDivergovalo as exc:
        print(f"  [CHYBA UCENI] {exc}")
    except Exception as exc:
        chyba_implementace(exc)


def faze_a2_polynom(cfg: ExperimentConfig) -> None:
    """Faze A2 -- kruznice: overeni kernelove triku, polynomialni jadro, kontrast s linearnim.

    Overeni triku je ve vlastnim ``try``, aby nedokonceny trenink nezahodil
    hotove overeni (a naopak).

    Parametry
    ---------
    cfg : ExperimentConfig
        Konfigurace experimentu.
    """
    banner("Faze A2: Polynomialni jadro -- kruznice (circles) a kernelovy trik")
    try:
        x, y = load_circles(cfg.data.n_samples, cfg.data.circles_noise,
                            cfg.data.circles_factor, cfg.seed)
        data = rozdel(x, y, "circles", cfg.data.test_size, cfg.seed)
    except Exception as exc:
        chyba_dat(exc)
        print("  -> Zkontrolujte sekci data v config.yaml; faze A2 se preskoci.")
        return
    graphs = cfg.paths.graphs_dir
    tol = cfg.diagnostics.trick_tolerance
    print(f"  data: trenink {tuple(data.x_train.shape)}, test {tuple(data.x_test.shape)}")

    # Kernelovy trik na vlastni oci: (a.b)^2 == phi(a).phi(b), phi do 3D.
    try:
        rozdil = verify_kernel_trick(PolynomialKernel(r=2), data.x_train, data.x_test)
        print("  kernelovy trik (r = 2): max |(x z^T)^2 - phi(x) phi(z)^T| = "
              f"{rozdil:.2e}")
        if rozdil < tol:
            print("  [OK] Jadro spocitane ve 2D se rovna skalarnimu soucinu ve 3D "
                  "prostoru priznaku -- phi jsme k treninku vubec nepotrebovali.")
        else:
            print(f"  [CHYBA] Rozdil presahuje diagnostics.trick_tolerance = {tol:.0e}; "
                  "zkontrolujte gram a feature_map.")
    except NotImplementedError as exc:
        faze_neni_hotova(exc)
    except Exception as exc:
        chyba_implementace(exc)

    try:
        kernel = jadro(cfg.kernels.circles)
        print(f"  jadro: {kernel!r}")
        vysledek = train_and_eval(tovarna_modelu(kernel, cfg.training), data,
                                  cfg.training.c, cfg.diagnostics)
        print(f"  presnost: trenink {vysledek.acc_tr:.3f}, test {vysledek.acc_te:.3f}; "
              f"support vektoru {vysledek.n_sv}")

        print("  kontrast: linearni jadro na tychz datech")
        linearni = train_and_eval(tovarna_modelu(LinearKernel(), cfg.training), data,
                                  cfg.training.c, cfg.diagnostics)
        print(f"  presnost linearniho jadra: test {linearni.acc_te:.3f} "
              "(primka kruznice neoddeli)")
        panely_png = os.path.join(graphs, "a2_kruznice_linear_vs_jadro.png")
        plot_boundary_panels(
            [(f"{LinearKernel()!r}: test {linearni.acc_te:.2f}",
              linearni.model.decision_function, linearni.model.sv_x_),
             (f"{kernel!r}: test {vysledek.acc_te:.2f}",
              vysledek.model.decision_function, vysledek.model.sv_x_)],
            data.x_train, data.y_train, panely_png, suptitle="A2: kruznice", ncols=2,
        )
        print(f"  graf ulozen: {panely_png}")
    except NotImplementedError as exc:
        faze_neni_hotova(exc)
    except UceniDivergovalo as exc:
        print(f"  [CHYBA UCENI] {exc}")
    except Exception as exc:
        chyba_implementace(exc)


def faze_a3_rbf(cfg: ExperimentConfig) -> None:
    """Faze A3 -- pulmesice: perceptron vs. linearni vs. jadrove SVM, mrizka c x gamma.

    Parametry
    ---------
    cfg : ExperimentConfig
        Konfigurace experimentu.
    """
    banner("Faze A3: RBF jadro -- pulmesice (moons), srovnani modelu a mrizka c x gamma")
    try:
        x, y = load_moons(cfg.data.n_samples, cfg.data.moons_noise, cfg.seed)
        data = rozdel(x, y, "moons", cfg.data.test_size, cfg.seed)
    except Exception as exc:
        chyba_dat(exc)
        print("  -> Zkontrolujte sekci data v config.yaml; faze A3 se preskoci.")
        return
    graphs = cfg.paths.graphs_dir
    tr = cfg.training
    print(f"  data: trenink {tuple(data.x_train.shape)}, test {tuple(data.x_test.shape)}")
    try:
        kernel = jadro(cfg.kernels.moons)
        print(f"  jadro: {kernel!r}")
        vysledek = train_and_eval(tovarna_modelu(kernel, tr), data, tr.c, cfg.diagnostics)
        linearni = train_and_eval(tovarna_modelu(LinearKernel(), tr), data, tr.c,
                                  cfg.diagnostics)
        perceptron_fn, perceptron_acc = nauc_perceptron(data, cfg.seed)
        print(f"  {'model':34s}{'test':>8s}{'SV':>6s}")
        print(f"  {'perceptron (cv8, sklearn)':34s}{perceptron_acc:8.3f}{'-':>6s}")
        print(f"  {'SVM ' + repr(LinearKernel()):34s}{linearni.acc_te:8.3f}{linearni.n_sv:6d}")
        print(f"  {'SVM ' + repr(kernel):34s}{vysledek.acc_te:8.3f}{vysledek.n_sv:6d}")
        srovnani_png = os.path.join(graphs, "a3_srovnani_modelu.png")
        plot_boundary_panels(
            [(f"perceptron: test {perceptron_acc:.2f}", perceptron_fn, None),
             (f"SVM {LinearKernel()!r}: test {linearni.acc_te:.2f}",
              linearni.model.decision_function, linearni.model.sv_x_),
             (f"SVM {kernel!r}: test {vysledek.acc_te:.2f}",
              vysledek.model.decision_function, vysledek.model.sv_x_)],
            data.x_train, data.y_train, srovnani_png,
            suptitle="A3: perceptron vs. linearni SVM vs. jadrove SVM", ncols=3,
        )
        print(f"  graf ulozen: {srovnani_png}")
    except NotImplementedError as exc:
        faze_neni_hotova(exc)
        return
    except UceniDivergovalo as exc:
        print(f"  [CHYBA UCENI] {exc}")
        return
    except Exception as exc:
        chyba_implementace(exc)
        return

    # Mrizka c x gamma: vzdy RBF jadro, hodnoty ze sekce sweep.
    try:
        panely = []
        print("  mrizka c x gamma (RBF), test presnost / pocet SV:")
        for gamma in cfg.sweep.grid_gamma_values:
            for c in cfg.sweep.grid_c_values:
                bunka = train_and_eval(tovarna_modelu(RBFKernel(gamma), tr), data, c,
                                       cfg.diagnostics, vypis_cas=False)
                print(f"    gamma = {gamma:<6g} c = {c:<6g}  test {bunka.acc_te:.3f}, "
                      f"SV {bunka.n_sv}")
                panely.append((f"gamma={gamma:g}, c={c:g}: test {bunka.acc_te:.2f}",
                               bunka.model.decision_function, bunka.model.sv_x_))
        mrizka_png = os.path.join(graphs, "a3_mrizka_c_gamma.png")
        plot_boundary_panels(panely, data.x_train, data.y_train, mrizka_png,
                             suptitle="A3: vliv c (sloupce) a gamma (radky), RBF jadro",
                             ncols=len(cfg.sweep.grid_c_values))
        print(f"  graf ulozen: {mrizka_png}")
    except NotImplementedError as exc:
        faze_neni_hotova(exc)
    except UceniDivergovalo as exc:
        print(f"  [CHYBA UCENI] mrizka c x gamma: {exc}")
    except Exception as exc:
        chyba_implementace(exc)


# ============================================================================ #
#  Cast B: breast cancer                                                       #
# ============================================================================ #
def faze_b1_breast_cancer(
    cfg: ExperimentConfig,
    ) -> tuple[DatasetBundle | None, TrainedModel | None, str]:
    """Faze B1 -- breast cancer: data, trenink s jadrem z config.yaml a metriky.

    Parametry
    ---------
    cfg : ExperimentConfig
        Konfigurace experimentu.

    Navratova hodnota
    -----------------
    tuple[DatasetBundle | None, TrainedModel | None, str]
        Data (nebo ``None``), natrenovany model (nebo ``None``) a duvod pro
        zavisle faze.
    """
    banner("Faze B1: Breast cancer -- 30 priznaku, jadro vybrane v config.yaml")
    try:
        x, y = load_breast_cancer()
        data = rozdel(x, y, "breast_cancer", cfg.data.test_size, cfg.seed)
    except Exception as exc:
        chyba_dat(exc)
        duvod = "Data breast cancer se nepodarilo pripravit (faze B1)."
        print("  -> Cela cast B se preskoci.")
        return None, None, duvod
    print(f"  data: trenink {tuple(data.x_train.shape)}, test {tuple(data.x_test.shape)}; "
          f"maligni (+1) v testu: {int((data.y_test > 0).sum())} z {len(data.y_test)}")
    duvod = "Model nebyl natrenovan (faze B1 neprobehla)."
    try:
        kernel = jadro(cfg.kernels.breast_cancer)
        print(f"  jadro: {kernel!r}; c = {cfg.training.c}")
        vysledek = train_and_eval(tovarna_modelu(kernel, cfg.training), data,
                                  cfg.training.c, cfg.diagnostics)
        print(f"  presnost: trenink {vysledek.acc_tr:.3f}, test {vysledek.acc_te:.3f}; "
              f"support vektoru {vysledek.n_sv} z {len(data.y_train)}")
        metriky = vyhodnot_klasifikaci(data.y_test, vysledek.model.predict(data.x_test))
        vypis_metriky(metriky, pozitivni="maligni", negativni="benigni")
        prubeh_png = os.path.join(cfg.paths.graphs_dir, "b1_prubeh_dualu.png")
        plot_dual_history(vysledek.model.history_, prubeh_png,
                          title=f"B1: breast cancer, {kernel!r}")
        print(f"  graf ulozen: {prubeh_png}")
        return data, vysledek, duvod
    except NotImplementedError as exc:
        faze_neni_hotova(exc)
    except UceniDivergovalo as exc:
        print(f"  [CHYBA UCENI] {exc}")
        duvod = "Trenink ve fazi B1 divergoval -- zmensete training.lr v config.yaml."
    except Exception as exc:
        chyba_implementace(exc)
    return data, None, duvod


def faze_b2_sweep(cfg: ExperimentConfig, data: DatasetBundle | None) -> None:
    """Faze B2 -- vliv regularizace c: sirka pasma, pocet SV, presnost.

    Parametry
    ---------
    cfg : ExperimentConfig
        Konfigurace experimentu.
    data : DatasetBundle | None
        Data z faze B1 (``None`` = data se nepodarilo pripravit).
    """
    banner("Faze B2: Vliv regularizace c (sweep_c) na breast cancer")
    if data is None:
        preskoceno("Data breast cancer nejsou k dispozici (faze B1).")
        return
    try:
        kernel = jadro(cfg.kernels.breast_cancer)
        print(f"  jadro: {kernel!r}; c z sweep.c_values = {cfg.sweep.c_values}")
        rows = sweep_c(cfg.sweep.c_values, data.x_train, data.y_train, data.x_test, data.y_test,
                       tovarna_modelu(kernel, cfg.training))
        radky = radky_sweepu(rows)
        print(f"  {'c':>9s}{'pasmo 2/||w||':>15s}{'SV':>6s}{'trenink':>9s}{'test':>8s}")
        for r in radky:
            print(f"  {r.c:9g}{r.margin:15.4f}{r.n_sv:6d}{r.acc_tr:9.3f}{r.acc_te:8.3f}")
        sweep_png = os.path.join(cfg.paths.graphs_dir, "b2_sweep_c.png")
        plot_c_sweep(rows, sweep_png, title=f"B2: vliv c, breast cancer, {kernel!r}")
        print(f"  graf ulozen: {sweep_png}")
        nejlepsi = max(radky, key=lambda r: r.acc_te)
        print(f"  nejvyssi testovaci presnost {nejlepsi.acc_te:.3f} pri c = {nejlepsi.c:g} "
              "(vyber podle testu je jen ilustrace -- spravne se ladi na validacnich datech)")
    except NotImplementedError as exc:
        faze_neni_hotova(exc)
    except KeyError as exc:
        print(f"  [CHYBA IMPLEMENTACE] sweep_c vratil slovnik bez klice {exc} "
              "(ocekavane klice: c, margin, n_sv, acc_train, acc_test).")
    except Exception as exc:
        chyba_implementace(exc)


def faze_b3_ulozeni(cfg: ExperimentConfig, data: DatasetBundle | None,
                    vysledek: TrainedModel | None, duvod: str) -> None:
    """Faze B3 -- ulozeni modelu z B1 do models/ a nacteni do nove instance.

    Parametry
    ---------
    cfg : ExperimentConfig
        Konfigurace experimentu.
    data : DatasetBundle | None
        Data z faze B1.
    vysledek : TrainedModel | None
        Model z faze B1 (``None`` = trenink neprobehl).
    duvod : str
        Proc model chybi (pro hlasku ``[PRESKOCENO]``).
    """
    banner("Faze B3: Ulozeni modelu a kontrolni nacteni")
    if data is None or vysledek is None:
        preskoceno(duvod)
        return
    tolerance = cfg.diagnostics.load_tolerance
    cesta = os.path.join(cfg.paths.models_dir, "svm_breast_cancer.pt")
    os.makedirs(cfg.paths.models_dir, exist_ok=True)
    try:
        vysledek.model.save(cesta)
        print(f"  model ulozen: {cesta} ({os.path.getsize(cesta) / 1024:.1f} kB, "
              f"{vysledek.n_sv} support vektoru)")
        nacteny = KernelSVM.load(cesta)
        print(f"  nacteno: {nacteny.kernel!r}, c = {nacteny.c}, "
              f"support vektoru {len(nacteny.sv_x_)}")
        rozdil = float((nacteny.decision_function(data.x_test)
                        - vysledek.model.decision_function(data.x_test)).abs().max())
        if rozdil < tolerance:
            print(f"  [OK] Nacteny model dava stejne h(x) (max. rozdil {rozdil:.1e}).")
        else:
            print(f"  [CHYBA] Nacteny model se lisi (max. rozdil {rozdil:.1e}, "
                  f"diagnostics.load_tolerance = {tolerance:.0e}).")
    except NotImplementedError as exc:
        faze_neni_hotova(exc)
    except pickle.UnpicklingError as exc:
        chyba_implementace(exc)
        print("  -> Ukladejte jen obycejny slovnik tenzoru, cisel a retezcu, ne cely objekt "
              "(torch.load s weights_only=True jiny obsah odmitne).")
    except Exception as exc:
        chyba_implementace(exc)


def main() -> None:
    """Spusti celou pipeline cviceni 12 s ochrannymi bloky u kazde faze."""
    banner("CVICENI 12 -- SVM a kernelovy trik (dual, PyTorch) -- start")

    # --- Config guard -----------------------------------------------------------
    try:
        cfg = load_config()
    except (ValueError, FileNotFoundError) as exc:
        print(f"[CHYBA KONFIGURACE] {exc}")
        sys.exit(1)

    torch.manual_seed(cfg.seed)
    log_path = priprav_logovani(cfg.paths.logs_dir)
    print(f"  prubeh kazdeho treninku (L(lambda), pocet SV) se zapisuje do {log_path}")

    # Cast A: 2D hracky
    faze_a1_linearni(cfg)
    faze_a2_polynom(cfg)
    faze_a3_rbf(cfg)

    # Cast B: breast cancer
    data_b, vysledek_b, duvod_b = faze_b1_breast_cancer(cfg)
    faze_b2_sweep(cfg, data_b)
    faze_b3_ulozeni(cfg, data_b, vysledek_b, duvod_b)

    banner("CVICENI 12 -- konec")


if __name__ == "__main__":
    main()
