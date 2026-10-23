# Papírové příklady — Cvičení 12

SVM a kernelový trik: šířka pásma, Gramovy matice jader, kernelový trik
na konkrétních vektorech, duální účelová funkce a krok projektovaného
gradientního výstupu, rozhodovací funkce se zahrnutým biasem a úvahy
o parametrech $C$ a $\gamma$. Příklady procvičují **přesně tu notaci a ty
výpočty**, se kterými pracujete v `src/kernels.py` a `src/svm.py`. Čísla jsou
volena tak, aby se dala spočítat na papíře.

**Řešení nejsou součástí repozitáře.** Výsledky si ověřte u vyučujícího nebo
výpočtem svými funkcemi ze `src/`, jakmile je doplníte (ukázka je na konci
souboru).

Značení je stejné jako v kódu:

- Body jsou řádky matice `x`, třídy $y_i\in\{-1,+1\}$.
- Lineární hranice $g(\mathbf{x}) = \mathbf{w}^\top\mathbf{x} + b$, šířka pásma
  $2/\lVert\mathbf{w}\rVert$, orientovaná vzdálenost bodu od hranice
  $z = g(\mathbf{x})/\lVert\mathbf{w}\rVert$.
- Jádra: lineární $K(\mathbf{a},\mathbf{b}) = \mathbf{a}^\top\mathbf{b}$,
  polynomiální $K(\mathbf{a},\mathbf{b}) = (\mathbf{a}^\top\mathbf{b})^r$,
  RBF $K(\mathbf{a},\mathbf{b}) = e^{-\gamma\lVert\mathbf{a}-\mathbf{b}\rVert^2}$.
- Gramova matice `gram(x, z)` má prvky $K_{ij} = K(\mathbf{x}_i, \mathbf{z}_j)$.
- Bias je zahrnut do jádra: $K_{\text{eff}} = K + 1$ (`_effective_gram`, v kódu `k_eff`).
- Duál: $L(\boldsymbol\lambda) = \sum_i\lambda_i - \tfrac12\sum_{i,j}\lambda_i\lambda_j y_i y_j K_{\text{eff},ij}$,
  maticově $L = \sum_i\lambda_i - \tfrac12\boldsymbol\lambda^\top Q\boldsymbol\lambda$,
  kde $Q = (\mathbf{y}\mathbf{y}^\top)\odot K_{\text{eff}}$ ($\odot$ = násobení po prvcích).
- Krok tréninku: $\boldsymbol\lambda \leftarrow \operatorname{clip}\big(\boldsymbol\lambda + \eta\,\partial L/\partial\boldsymbol\lambda,\ 0,\ C\big)$.
- Rozhodovací funkce: $h(\mathbf{x}) = \sum_{i\in SV}\lambda_i y_i\big(K(\mathbf{x}_i,\mathbf{x}) + 1\big)$,
  třída podle znaménka ($h \ge 0 \Rightarrow +1$); ekvivalentně
  $h(\mathbf{x}) = \sum_{i\in SV}\lambda_i y_i K(\mathbf{x}_i,\mathbf{x}) + b$ s $b = \sum_{i\in SV}\lambda_i y_i$.

---

## Příklad 1 — Hranice, pásmo a vzdálenost bodů

Lineární SVM ve 2D našlo hranici s $\mathbf{w} = (3,\ 4)$ a $b = -5$.

**Úkoly:**

- **a)** Spočítejte $\lVert\mathbf{w}\rVert$ a šířku pásma.
- **b)** Pro body $A = (1, 1)$, $B = (1;\ 0{,}5)$, $C = (2, 0)$, $D = (0, 1)$,
  $E = (0, 0)$ a $F = (2, 1)$ spočítejte $g(\mathbf{x})$ a orientovanou
  vzdálenost $z$. Do které třídy každý bod patří?
- **c)** Které z bodů leží přesně na hranici a které přesně na okraji pásma?
  Které z nich mohou být support vektory?
- **d)** Vynásobte $\mathbf{w}$ i $b$ dvěma. Změní se hranice? Změní se
  vzdálenosti $z$? Změní se hodnoty $g$? Vysvětlete, proč se při odvození
  SVM požaduje $g(\mathbf{x}') = \pm 1$ pro nejbližší body.

---

## Příklad 2 — Gramovy matice tří bodů

Jsou dány body $\mathbf{x}_1 = (1, 0)$, $\mathbf{x}_2 = (0, 1)$,
$\mathbf{x}_3 = (1, 1)$, tedy `x = [[1, 0], [0, 1], [1, 1]]`.

**Úkoly:**

- **a)** Spočítejte Gramovu matici `gram(x, x)` lineárního jádra (matice
  $3\times 3$).
- **b)** Spočítejte Gramovu matici polynomiálního jádra s $r = 2$. Ověřte, že
  vznikne umocněním matice z bodu a) **po prvcích**. Čím by se lišil
  výsledek maticového součinu $K\cdot K$?
- **c)** Spočítejte matici čtverců vzdáleností $\lVert\mathbf{x}_i - \mathbf{x}_j\rVert^2$
  a z ní Gramovu matici RBF jádra pro $\gamma = \ln 2$. (Vodítko:
  $e^{-d^2\ln 2} = 2^{-d^2}$, kde $d^2$ je čtverec vzdálenosti.)
- **d)** Ověřte na matici RBF z bodu c): je symetrická? Co je na diagonále
  a proč? Který pár bodů je si nejméně podobný?
- **e)** Jak by vypadala RBF matice pro $\gamma\to\infty$ a pro $\gamma\to 0$?
  Co by to znamenalo pro schopnost SVM rozlišovat body?
- **f)** Zapište matici $K_{\text{eff}} = K + 1$ pro lineární jádro z bodu a).

---

## Příklad 3 — Kernelový trik na vlastní oči

Pro polynomiální jádro stupně 2 ve 2D platí
$\varphi(\mathbf{x}) = (x_1^2,\ \sqrt2\,x_1x_2,\ x_2^2)$.

**Úkoly:**

- **a)** Pro $\mathbf{a} = (1, 2)$ a $\mathbf{b} = (3, -1)$ spočítejte
  $K(\mathbf{a},\mathbf{b}) = (\mathbf{a}^\top\mathbf{b})^2$.
- **b)** Pro tytéž body spočítejte $\varphi(\mathbf{a})$, $\varphi(\mathbf{b})$
  a jejich skalární součin. Shoduje se s výsledkem a)?
- **c)** Totéž pro $\mathbf{a} = (2, 1)$ a $\mathbf{b} = (1, 3)$.
- **d)** Spočítejte, kolik násobení potřebuje každá z obou cest (jádro
  přímo vs. přes $\varphi$).
- **e)** Pro **tři** příznaky a $r = 2$ vypište složky $\varphi(\mathbf{x})$
  (vodítko: rozepište $(a_1b_1 + a_2b_2 + a_3b_3)^2$). Kolik jich je?
- **f)** Proč pro RBF jádro nelze podobný výpočet přes $\varphi$ provést?

---

## Příklad 4 — Duál, gradient a krok projektovaného výstupu

Dva trénovací body na ose $x_1$: $\mathbf{x}_1 = (1, 0)$ s $y_1 = +1$
a $\mathbf{x}_2 = (-1, 0)$ s $y_2 = -1$. Jádro je lineární.

**Úkoly:**

- **a)** Spočítejte $K$, $K_{\text{eff}} = K + 1$ a $Q = (\mathbf{y}\mathbf{y}^\top)\odot K_{\text{eff}}$.
- **b)** Zapište $L(\lambda_1, \lambda_2)$ jako funkci dvou proměnných
  a spočítejte $L(0, 0)$ a $L(0{,}3;\ 0{,}2)$.
- **c)** Spočítejte gradient $\partial L/\partial\boldsymbol\lambda = \mathbf{1} - Q\boldsymbol\lambda$
  v bodech $(0, 0)$ a $(0{,}3;\ 0{,}2)$.
- **d)** Proveďte jeden krok z $\boldsymbol\lambda = (0, 0)$ s $\eta = 0{,}1$
  a $C = 1$. Vzrostlo $L$?
- **e)** Proveďte jeden krok z $\boldsymbol\lambda = (0{,}3;\ 0{,}2)$
  s $\eta = 0{,}5$ a $C = 0{,}4$. Jaká je hodnota před projekcí a jaká po ní?
  Je projekce aktivní?
- **f)** Najděte maximum $L$ pro $C \ge 0{,}5$ (položte gradient roven nule).
  Spočítejte $\mathbf{w} = \sum_i\lambda_i y_i\mathbf{x}_i$, $b = \sum_i\lambda_i y_i$
  a šířku pásma. Odpovídá geometrii obou bodů?
- **g)** Rozhodovací funkcí $h(\mathbf{x})$ s optimálními $\lambda$ zařaďte
  body $(0{,}5;\ 3)$ a $(-2, 1)$.

---

## Příklad 5 — Klasifikace nového bodu RBF jádrem

Natrénované RBF SVM s $\gamma = \ln 2$ má tři support vektory:

| $i$ | $\mathbf{x}_i$ | $y_i$ | $\lambda_i$ |
|:---:|:---:|:---:|:---:|
| 1 | $(0, 0)$ | $+1$ | $0{,}5$ |
| 2 | $(2, 0)$ | $-1$ | $0{,}25$ |
| 3 | $(0, 2)$ | $-1$ | $0{,}25$ |

**Úkoly:**

- **a)** Spočítejte práh $b = \sum_i\lambda_i y_i$.
- **b)** Pro bod $\mathbf{p} = (1, 0)$ spočítejte čtverce vzdáleností ke
  všem support vektorům, hodnoty jádra, $h(\mathbf{p})$ a třídu.
- **c)** Totéž pro bod $\mathbf{q} = (2, 1)$.
- **d)** Totéž pro bod $(0, 0)$, který je sám support vektorem. Je zařazen
  správně? Leží mimo pásmo, na jeho okraji, nebo uvnitř ($|h| < 1$)?
- **e)** Kam se blíží $h(\mathbf{x})$ pro bod velmi vzdálený od všech
  support vektorů, např. $(10, 10)$? Jak spolehlivá je predikce RBF SVM daleko
  od trénovacích dat?

---

## Příklad 6 — Úvahy o $C$, $\gamma$ a support vektorech

**Úkoly:**

- **a)** Po tréninku s $C = 1$ vyšlo $\boldsymbol\lambda = (0;\ 0{,}3;\ 1;\ 0;\ 1;\ 0{,}7)$.
  Které body jsou support vektory? Které leží přesně na okraji pásma a které
  uvnitř pásma nebo na nesprávné straně?
- **b)** Bod 4 ($\lambda_4 = 0$) z trénovacích dat odstraníme a model
  natrénujeme znovu. Změní se hranice? A při odstranění bodu 3?
- **c)** Model má 398 support vektorů z 398 trénovacích bodů, trénovací
  přesnost 1,000 a testovací 0,63. Která z hodnot $C$, $\gamma$ je
  pravděpodobně příliš velká a proč?
- **d)** Při zvětšování $C$ od 0,01 do 1 se šířka pásma zmenšila a počet
  support vektorů klesl. Vysvětlete obojí pomocí boxu $0\le\lambda_i\le C$.
- **e)** Krok tréninku `lr` byl zvětšen desetkrát a $L(\boldsymbol\lambda)$
  během tréninku místo růstu kleslo do záporných hodnot. Co se stalo?
  Jak souvisí stabilní krok s největším vlastním číslem matice $Q$?
- **f)** Proč se před tréninkem SVM příznaky standardizují? Co by se stalo
  s RBF jádrem, kdyby jeden příznak byl v mm a ostatní v m?

---

## Ověření výsledků vlastním kódem

Po dokončení bloků I–V můžete výsledky ověřit (spouštějte z kořene
repozitáře):

```python
import math
import torch
from src import KernelSVM, LinearKernel, PolynomialKernel, RBFKernel

# Příklad 2
x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float64)
print(LinearKernel().gram(x, x))
print(PolynomialKernel(2).gram(x, x))
print(RBFKernel(math.log(2)).gram(x, x))

# Příklad 3
a = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
b = torch.tensor([[3.0, -1.0]], dtype=torch.float64)
k = PolynomialKernel(2)
print(k.gram(a, b), k.feature_map(a) @ k.feature_map(b).T)

# Příklad 4: jeden krok a duální funkce
x4 = torch.tensor([[1.0, 0.0], [-1.0, 0.0]], dtype=torch.float64)
y4 = torch.tensor([1.0, -1.0], dtype=torch.float64)
svm = KernelSVM(LinearKernel(), c=0.4, lr=0.5, n_iters=1)
k_eff = svm._effective_gram(x4, x4)
lam = torch.tensor([0.3, 0.2], dtype=torch.float64)
print(svm.dual_objective(lam, k_eff, y4), svm._ascent_step(lam, k_eff, y4))

# Příklad 5: rozhodovací funkce ze zadaných support vektorů
svm5 = KernelSVM(RBFKernel(math.log(2)), c=1.0, lr=0.1, n_iters=1)
svm5.sv_x_ = torch.tensor([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]], dtype=torch.float64)
svm5.sv_y_ = torch.tensor([1.0, -1.0, -1.0], dtype=torch.float64)
svm5.sv_lambdas_ = torch.tensor([0.5, 0.25, 0.25], dtype=torch.float64)
points = torch.tensor([[1.0, 0.0], [2.0, 1.0], [0.0, 0.0]], dtype=torch.float64)
print(svm5.decision_function(points), svm5.predict(points))
```
