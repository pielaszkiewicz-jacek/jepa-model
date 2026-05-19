# 📐 Model Matematyczny R-JEPA

> Szczegółowy opis matematyczny modelu Recurrent Joint Embedding Predictive Architecture.

---

## Spis treści

1. [Notacja](#1-notacja)
2. [Cel modelu](#2-cel-modelu)
3. [Enkoder czasowy](#3-enkoder-czasowy)
4. [Momentum Encoder](#4-momentum-encoder)
5. [Predyktor rekurencyjny](#5-predyktor-rekurencyjny)
6. [Funkcja straty JEPA](#6-funkcja-straty-jepa)
7. [Regularacja](#7-regularyzacja)
8. [Dekoder](#8-dekoder)
9. [Predykcja ensemble](#9-predykcja-ensemble)
10. [Pełny przepływ treningowy](#10-pełny-przepływ-treningowy)

---

## 1. Notacja

| Symbol | Wymiar | Opis |
|--------|--------|------|
| $x_t$ | $\mathbb{R}^d$ | Obserwacja w chwili $t$ ($d$ cech) |
| $X_c = [x_{t-T+1}, \dots, x_t]$ | $\mathbb{R}^{T \times d}$ | Okno kontekstowe ($T$ kroków) |
| $X_t = [x_{t+1}, \dots, x_{t+H}]$ | $\mathbb{R}^{H \times d}$ | Okno targetu ($H$ kroków) |
| $z_c \in \mathbb{R}^L$ | $\mathbb{R}^L$ | Latent kontekstu (enkoder online) |
| $z_t \in \mathbb{R}^L$ | $\mathbb{R}^L$ | Latent targetu (momentum encoder) |
| $\hat{z}_{t+k} \in \mathbb{R}^L$ | $\mathbb{R}^L$ | Przewidywany latent w kroku $k$ |
| $Z_c = [z_{t-T+1}, \dots, z_t]$ | $\mathbb{R}^{T \times L}$ | Sekwencja latentów kontekstu |
| $\theta$ | — | Parametry enkodera online |
| $\theta_m$ | — | Parametry momentum enkodera |
| $\tau$ | $(0,1)$ | Współczynnik EMA (momentum) |

---

## 2. Cel modelu

Celem R-JEPA jest nauczenie się takiej przestrzeni latentnej $Z \subset \mathbb{R}^L$, w której:

1. **Latenty są predykcyjne** — latent kontekstu $z_c$ pozwala przewidzieć latent targetu $z_t$
2. **Latenty są bogate informacyjnie** — nie dochodzi do collapse'u (wszystkie wymiary są wykorzystane)
3. **Latenty są dekorrelowane** — każdy wymiar niesie niezależną informację

**Główna idea JEPA**: Zamiast przewidywać $x_{t+1}$ z $x_t$ (co jest trudne ze względu na wysoką wymiarowość i szum), przewidujemy $z_{t+1}$ z $z_t$ w uproszczonej przestrzeni latentnej.

```
Cel:  min  d( f_online(X_c),  f_momentum(X_t) )
           \______________/  \_______________/
          przewidywany        rzeczywisty
          latent targetu      latent targetu
```

gdzie $d(\cdot, \cdot)$ to miara odległości w przestrzeni latentnej.

---

## 3. Enkoder czasowy

Enkoder $f_\theta: \mathbb{R}^{T \times d} \to \mathbb{R}^L$ transformuje okno czasowe do przestrzeni latentnej.

### 3.1 Architektura

```
Wejście: x ∈ ℝ^{T×d}
    │
    ├── Conv1D(k=3) → BatchNorm → GELU      (lokalne wzorce)
    ├── Conv1D(k=3) → BatchNorm → GELU
    │
    ├── BiLSTM(hidden=h, layers=N)           (zależności czasowe)
    │
    ├── concat(forward_last, backward_last)
    │
    ├── Linear(2h → h) → LayerNorm → GELU   (projekcja)
    ├── Dropout(p=0.1)
    └── Linear(h → L) → LayerNorm           (latent)

Wyjście: z ∈ ℝ^L
```

### 3.2 Konwolucje 1D

Dla wejścia $X \in \mathbb{R}^{T \times d}$:

$$h^{(1)}_t = \text{GELU}\left(\text{BN}\left(W^{(1)} * X_{t-1:t+1} + b^{(1)}\right)\right)$$

$$h^{(2)}_t = \text{GELU}\left(\text{BN}\left(W^{(2)} * h^{(1)}_{t-1:t+1} + b^{(2)}\right)\right)$$

gdzie $*$ oznacza splot 1D z jądrem $k=3$, a $\text{BN}$ to Batch Normalization.

### 3.3 BiLSTM

Dla sekwencji $h^{(2)}_1, \dots, h^{(2)}_T$:

$$\overrightarrow{h}_t = \text{LSTM}_{\text{fwd}}(h^{(2)}_t, \overrightarrow{h}_{t-1})$$
$$\overleftarrow{h}_t = \text{LSTM}_{\text{bwd}}(h^{(2)}_t, \overleftarrow{h}_{t+1})$$

$$h^{\text{lstm}}_t = [\overrightarrow{h}_t; \overleftarrow{h}_t] \in \mathbb{R}^{2h}$$

### 3.4 Projekcja do latent space

Ostatnia warstwa łączy forward i backward stany ukryte:

$$z = W_{\text{proj}} \cdot [\overrightarrow{h}_T; \overleftarrow{h}_1] + b_{\text{proj}}$$

Alternatywnie, dla całej sekwencji:

$$z_t = W_{\text{proj}} \cdot h^{\text{lstm}}_t + b_{\text{proj}} \quad \text{(return\_sequence=True)}$$

---

## 4. Momentum Encoder

Momentum encoder $f_{\theta_m}$ to kopia enkodera online aktualizowana przez EMA.

### 4.1 Aktualizacja EMA

Parametry momentum enkodera są aktualizowane po każdym kroku optymalizacji:

$$\theta_m \leftarrow \tau \cdot \theta_m + (1 - \tau) \cdot \theta$$

gdzie:
- $\tau = 0.996$ — współczynnik momentum (bliski 1 dla wolnej aktualizacji)
- $\theta$ — parametry enkodera online (aktualizowane przez gradient)
- $\theta_m$ — parametry momentum enkodera (zamrożone, bez gradientu)

### 4.2 Własności

1. **Stabilne targety**: Momentum encoder zmienia się wolniej niż enkoder online, co zapewnia stabilne cele do uczenia
2. **Bez gradientu**: $\nabla_{\theta_m} \mathcal{L} = 0$ — oszczędność pamięci i obliczeń
3. **Wolna ewolucja**: Dzięki $\tau \approx 1$, reprezentacje zmieniają się płynnie

---

## 5. Predyktor rekurencyjny

Predyktor $g_\phi: \mathbb{R}^L \to \mathbb{R}^{H \times L}$ przekształca latent kontekstu w sekwencję przyszłych latentów.

### 5.1 Architektura

Predyktor obsługuje **dwa scenariusze inicjalizacji** GRU:

1. **Sekwencja kontekstu dostępna** ($Z_c \in \mathbb{R}^{T \times L}$): GRU przetwarza całą sekwencję, ostatni stan ukryty staje się inicjalizacją autoregresji.
2. **Tylko pojedynczy latent** ($z_c \in \mathbb{R}^L$): `context_aggregator` projektuje $L \to H_{\text{hidden}}$, a wynik jest używany jako stan początkowy GRU.

```
              ┌─── Z_c ∈ ℝ^{T×L} ──→ GRU(seq) ──→ h_init
              │
Wejście: z_c ─┤                    ┌─ Linear(L→H) ─┐
              │                    ├─ LayerNorm    ├─→ h_init ∈ ℝ^{H}
              └─── (sam latent) ──→└─ GELU        ┘
                                        (context_aggregator)

GRU(input_size=L, hidden_size=H, N layers)
    │
    ├── Dla k = 1..H:
    │   ├── input:  ẑ_{k-1} ∈ ℝ^L     (input_size = L)
    │   ├── GRU_step(ẑ_{k-1}, h_{k-1})
    │   ├── Proj: Linear(H → H) → LN → GELU → Dropout → Linear(H → L)
    │   └── ẑ_k ∈ ℝ^L
    │
    └── Stack: [ẑ_1, ẑ_2, ..., ẑ_H] ∈ ℝ^{H×L}

Wyjście: Ẑ ∈ ℝ^{H×L}
```

**Uwaga**: GRU ma `input_size = latent_dim = L` — zarówno wejście inicjalizacyjne (przez `context_aggregator` jako stan ukryty, nie jako wejście GRU), jak i każdy krok autoregresji operują na wektorach wymiaru $L$.

### 5.2 Proces autoregresyjny

Predykcja latentów odbywa się krok po kroku.

**Inicjalizacja** (dwa warianty):

Gdy dostępna jest pełna sekwencja kontekstu $Z_c \in \mathbb{R}^{T \times L}$:
$$h_0 = \text{GRU}_{\text{seq}}(Z_c) \quad \text{(ostatni stan ukryty po przetworzeniu całej sekwencji)}$$

Gdy dostępny jest tylko pojedynczy latent $z_c \in \mathbb{R}^L$:
$$h_0 = \text{Aggregator}(z_c) = \text{GELU}(\text{LayerNorm}(W_{\text{agg}} \cdot z_c + b_{\text{agg}}))$$
gdzie $W_{\text{agg}} \in \mathbb{R}^{H \times L}$.

**Pętla autoregresyjna** (jednakowa dla obu wariantów):

Dla $k = 1, \dots, H$:

$$h_k = \text{GRU}(\hat{z}_{k-1}, h_{k-1})$$
$$\hat{z}_k = W_{\text{out},2} \cdot \text{GELU}(\text{LayerNorm}(W_{\text{out},1} \cdot h_k + b_{\text{out},1})) + b_{\text{out},2}$$

gdzie $\hat{z}_0 = z_c$ (latent kontekstu jako pierwsze "przewidywanie"), a $W_{\text{out},1} \in \mathbb{R}^{H \times H}$, $W_{\text{out},2} \in \mathbb{R}^{L \times H}$.

### 5.3 Predykcja z szumem (ensemble)

Dla estymacji niepewności, dodajemy szum do każdego kroku:

$$\tilde{z}_{k-1} = \hat{z}_{k-1} + \epsilon \cdot \eta_{k-1}, \quad \eta_{k-1} \sim \mathcal{N}(0, I)$$

$$h_k = \text{GRU}(\tilde{z}_{k-1}, h_{k-1})$$

Powtarzając ten proces $N$ razy ($N$ = ensemble_size), otrzymujemy $N$ trajektorii predykcyjnych.

---

## 6. Funkcja straty JEPA

Funkcja straty JEPA łączy trzy komponenty:

$$\mathcal{L}_{\text{JEPA}} = \mathcal{L}_{\text{pred}} + \lambda_{\text{var}} \cdot \mathcal{L}_{\text{var}} + \lambda_{\text{cov}} \cdot \mathcal{L}_{\text{cov}}$$

gdzie $\lambda_{\text{var}} = 0.5$, $\lambda_{\text{cov}} = 0.1$.

### 6.1 Prediction Loss (MSE w latent space)

$$\mathcal{L}_{\text{pred}} = \frac{1}{2} \cdot \text{MSE}(\bar{\hat{z}}, z_t) + \frac{1}{2} \cdot \frac{1}{H} \sum_{k=1}^{H} \text{MSE}(\hat{z}_k, z_t)$$

gdzie $\bar{\hat{z}} = \frac{1}{H} \sum_{k=1}^{H} \hat{z}_k$ to średni przewidywany latent.

Pierwszy człon to MSE średniej predykcji, drugi to MSE krok po kroku.

**Dlaczego MSE w latent space?** Ponieważ przewidywanie w przestrzeni latentnej jest łatwiejsze niż w przestrzeni obserwacji — latenty mają niższy wymiar i są pozbawione szumu.

### 6.2 Variance Regularization

$$\mathcal{L}_{\text{var}} = \frac{1}{2} \left[ \frac{1}{L} \sum_{j=1}^{L} \max(0, \epsilon - \text{Std}(\bar{\hat{z}}_j)) + \frac{1}{L} \sum_{j=1}^{L} \max(0, \epsilon - \text{Std}(z_{t,j})) \right]$$

gdzie:
- $\text{Std}(\cdot)$ to odchylenie standardowe w obrębie batcha
- $\epsilon = 0.001$ to minimalny próg wariancji
- $\max(0, \epsilon - \text{Std})$ to funkcja hinge loss

**Cel**: Zapobiega collapse'owi — wymusza, aby każdy wymiar latentu miał wariancję >= $\epsilon$.

### 6.3 Covariance Regularization

$$\mathcal{L}_{\text{cov}} = \frac{1}{2L} \sum_{i \neq j} [\text{Cov}(\bar{\hat{z}})]_{ij}^2 + \frac{1}{2L} \sum_{i \neq j} [\text{Cov}(z_t)]_{ij}^2$$

gdzie $\text{Cov}(z) = \frac{1}{B-1} (z - \bar{z})^T (z - \bar{z})$ to macierz kowariancji w obrębie batcha.

**Cel**: Dekorreluje wymiary latentów — każdy wymiar niesie niezależną informację.

---

## 7. Regularyzacja — wizualizacja

```
🔴 Problem: Collapse (wszystkie latenty takie same)
    z₁, z₂, ..., z_B → wszystkie ≈ c
    Var(z) ≈ 0, Cov(z) ≈ 0
    → model nie nauczony

🟢 Rozwiązanie: Variance + Covariance regularization
    Var(z) ≥ ε (każdy wymiar aktywny)
    Cov(z) ≈ I (wymiary dekorrelowane)
    → model nauczony
```

---

## 8. Dekoder

Dekoder $h_\psi: \mathbb{R}^L \to \mathbb{R}^d$ przekształca latenty z powrotem do przestrzeni cen.

### 8.1 Architektura

```
Wejście: z ∈ ℝ^L
    │
    ├── Linear(L → 64) → LayerNorm → GELU
    ├── Linear(64 → 64) → LayerNorm → GELU
    └── Linear(64 → d)

Wyjście: x̂ ∈ ℝ^d
```

### 8.2 Zastosowanie

Dekoder jest używany głównie podczas **wnioskowania** — do konwersji przewidywanych latentów na ceny akcji:

$$\hat{x}_{t+k} = h_\psi(\hat{z}_{t+k}), \quad k = 1, \dots, H$$

Podczas treningu dekoder nie uczestniczy w stracie — strata operuje wyłącznie w przestrzeni latentnej.

---

## 9. Predykcja ensemble

### 9.1 Algorytm

```
Input:  context X_c ∈ ℝ^{T×d}
        ensemble_size N
        noise_scale ε
        horizon H

1. z_c, Z_c = f_online(X_c)
2. For i = 1..N:
3.     ẑ^{(i)}_1 = GRU(z_c) + ε·η_1
4.     ẑ^{(i)}_2 = GRU(ẑ^{(i)}_1) + ε·η_2
5.     ...
6.     ẑ^{(i)}_H = GRU(ẑ^{(i)}_{H-1}) + ε·η_H
7.     X̂^{(i)} = h_ψ(ẑ^{(i)})
8. Mean = (1/N) · Σ_i X̂^{(i)}
9. Std = sqrt((1/N) · Σ_i (X̂^{(i)} - Mean)²)
10. CI_upper = Mean + 1.96 · Std
11. CI_lower = Mean - 1.96 · Std
```

### 9.2 Przedział ufności

Dla poziomu ufności 95%:

$$\text{CI} = \hat{x} \pm 1.96 \cdot \hat{\sigma}$$

gdzie $\hat{\sigma}$ to odchylenie standardowe ensemble.

---

## 10. Pełny przepływ treningowy

### Algorytm treningu R-JEPA

```
1. Pobierz batch: (X_c, X_t) ~ DataLoader
2. 
3. # Forward pass (enkoder online)
4. z_c = f_online(X_c)                    # latent kontekstu
5. Z_c = f_online(X_c, return_sequence)   # sekwencja latentów
6.
7. # Forward pass (momentum encoder — bez gradientu)
8. z_t = f_momentum(X_t)                  # latent targetu
9.
10. # Predykcja w latent space
11. Ẑ = g_ϕ(z_c, H, Z_c)                  # przewidywane latenty
12.
13. # Oblicz stratę JEPA
14. L_pred = MSE(Ẑ, z_t)                  # prediction loss
15. L_var = variance_reg(Ẑ, z_t)          # variance regularization
16. L_cov = covariance_reg(Ẑ, z_t)        # covariance regularization
17. L = L_pred + λ_var·L_var + λ_cov·L_cov
18.
19. # Backward pass
20. L.backward()
21. clip_grad_norm(θ, max_norm=1.0)
22. optimizer.step()
23.
24. # Update momentum encoder
25. θ_m ← τ·θ_m + (1-τ)·θ
```

---

## Podsumowanie

R-JEPA to model samonadzorowany, który:

1. **Enkoduje** okna czasowe do przestrzeni latentnej (enkoder online + momentum)
2. **Przewiduje** przyszłe latenty za pomocą rekurencyjnego predyktora (GRU)
3. **Regularyzuje** reprezentacje przez variance + covariance loss
4. **Dekoduje** latenty z powrotem do cen podczas wnioskowania
5. **Estymuje niepewność** przez ensemble z szumem

Kluczowa innowacja: **predykcja w latent space** zamiast w observation space, co czyni uczenie bardziej efektywnym i odpornym na szum.
