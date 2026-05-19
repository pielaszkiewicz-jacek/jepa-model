# 🔧 Strojenie R-JEPA — dobór parametrów do sprzętu i danych

> Jak skonfigurować model R-JEPA, aby działał optymalnie na posiadanym sprzęcie
> i dla konkretnej charakterystyki danych rynkowych.

---

## Spis treści

- [🔧 Strojenie R-JEPA — dobór parametrów do sprzętu i danych](#-strojenie-r-jepa--dobór-parametrów-do-sprzętu-i-danych)
  - [Spis treści](#spis-treści)
  - [1. Wprowadzenie — co można stroić?](#1-wprowadzenie--co-można-stroić)
  - [2. Profile sprzętowe](#2-profile-sprzętowe)
    - [2.1 CPU-only (brak GPU)](#21-cpu-only-brak-gpu)
    - [2.2 GPU Low VRAM (4–8 GB)](#22-gpu-low-vram-48-gb)
    - [2.3 GPU Medium VRAM (12–16 GB)](#23-gpu-medium-vram-1216-gb)
    - [2.4 GPU High VRAM (24+ GB)](#24-gpu-high-vram-24-gb)
    - [2.5 GPU RTX 50-series (Blackwell, GDDR7)](#25-gpu-rtx-50-series-blackwell-gddr7)
      - [RTX 5090 (32 GB) — flagowa konfiguracja](#rtx-5090-32-gb--flagowa-konfiguracja)
      - [RTX 5080 (16 GB) — wysoka średnia półka](#rtx-5080-16-gb--wysoka-średnia-półka)
      - [RTX 5070 Ti / 5070 (12–16 GB) — średnia półka](#rtx-5070-ti--5070-1216-gb--średnia-półka)
      - [RTX 5060 (8 GB) — budżetowa Blackwell](#rtx-5060-8-gb--budżetowa-blackwell)
  - [3. Profile danych](#3-profile-danych)
    - [3.1 Dane daily (domyślne)](#31-dane-daily-domyślne)
    - [3.2 Dane intraday — 1 godzina](#32-dane-intraday--1-godzina)
    - [3.3 Dane intraday — 1 minuta](#33-dane-intraday--1-minuta)
    - [3.4 Dane o wysokiej rozdzielczości (1s / 100ms)](#34-dane-o-wysokiej-rozdzielczości-1s--100ms)
    - [3.5 Wiele tickerów jednocześnie](#35-wiele-tickerów-jednocześnie)
  - [4. Szczegółowy opis parametrów](#4-szczegółowy-opis-parametrów)
    - [4.1 Parametry danych](#41-parametry-danych)
    - [4.2 Parametry enkodera](#42-parametry-enkodera)
    - [4.3 Parametry predyktora](#43-parametry-predyktora)
    - [4.4 Parametry dekodera](#44-parametry-dekodera)
    - [4.5 Parametry JEPA](#45-parametry-jepa)
    - [4.6 Parametry treningu](#46-parametry-treningu)
  - [5. Macierz zależności sprzęt–parametry](#5-macierz-zależności-sprzętparametry)
  - [6. Drzewo decyzyjne — jak dobrać konfigurację](#6-drzewo-decyzyjne--jak-dobrać-konfigurację)
  - [7. Optymalizacja wydajności](#7-optymalizacja-wydajności)
    - [7.1 Automatic Mixed Precision (AMP)](#71-automatic-mixed-precision-amp)
    - [7.2 Gradient Accumulation](#72-gradient-accumulation)
    - [7.3 Gradient Checkpointing](#73-gradient-checkpointing)
    - [7.4 Batch Size tuning](#74-batch-size-tuning)
    - [7.5 num\_workers i pin\_memory](#75-num_workers-i-pin_memory)
  - [8. Przepisy — gotowe konfiguracje YAML](#8-przepisy--gotowe-konfiguracje-yaml)
    - [8.1 Lekka — CPU / laptop](#81-lekka--cpu--laptop)
    - [8.2 Standard — RTX 3060/4070, daily](#82-standard--rtx-30604070-daily)
    - [8.3 Szybka — RTX 4090, daily](#83-szybka--rtx-4090-daily)
    - [8.4 Wielotickerowa — RTX 4090, 10+ tickerów](#84-wielotickerowa--rtx-4090-10-tickerów)
    - [8.5 High-frequency — A100, 1-minuta](#85-high-frequency--a100-1-minuta)
    - [8.6 RTX 5090 — high-frequency, Transformer](#86-rtx-5090--high-frequency-transformer)
  - [9. Monitorowanie i diagnostyka](#9-monitorowanie-i-diagnostyka)
    - [9.1 Sprawdzanie wykorzystania VRAM](#91-sprawdzanie-wykorzystania-vram)
    - [9.2 Monitorowanie podczas treningu](#92-monitorowanie-podczas-treningu)
    - [9.3 Sygnały ostrzegawcze](#93-sygnały-ostrzegawcze)

---

## 1. Wprowadzenie — co można stroić?

R-JEPA oferuje wiele parametrów wpływających na balans **szybkość ↔ pamięć ↔ jakość predykcji**:

| Warstwa | Parametr | Wpływ na VRAM | Wpływ na szybkość | Wpływ na jakość |
|---------|----------|:---:|:---:|:---:|
| **Dane** | `sequence_length` | ⬆️ liniowy | ⬇️ liniowy | ⬆️ większy kontekst |
| **Dane** | `prediction_horizon` | ⬆️ liniowy | ⬇️ liniowy | ⬇️ dalsze horyzonty = trudniejsze |
| **Dane** | `batch_size` | ⬆️ liniowy | ⬆️ (przy GPU) | ⬆️ stabilniejsze gradienty |
| **Encoder** | `encoder_hidden_dim` | ⬆️ kwadratowy | ⬇️ | ⬆️ większa pojemność |
| **Encoder** | `encoder_num_layers` | ⬆️ liniowy | ⬇️ | ⬆️ głębsze reprezentacje |
| **Encoder** | `latent_dim` | ⬆️ liniowy | ⬇️ | ⬆️ kluczowy dla jakości |
| **Predictor** | `predictor_hidden_dim` | ⬆️ kwadratowy | ⬇️ | ⬆️ lepsze przewidywania |
| **Predictor** | `predictor_num_layers` | ⬆️ liniowy | ⬇️ | ⬆️ |
| **Predictor** | `predictor_dropout` | — | — | ⬆️ regularyzacja (przeuczenie) |
| **Decoder** | `decoder_hidden_dim` | ⬆️ | ⬇️ | ⬆️ lepsza rekonstrukcja |
| **JEPA** | `momentum_encoder_tau` | — | — | stabilność treningu |
| **Training** | `num_epochs` | — | ⬇️ liniowy | ⬆️ do pewnego momentu |
| **Training** | `learning_rate` | — | — | kluczowy dla konwergencji |
| **Training** | `weight_decay` | — | — | regularyzacja |
| **Training** | `use_amp` | ⬇️ ~40% | ⬆️ ~2x | minimalny (precyzja fp16) |

---

## 2. Profile sprzętowe

### 2.1 CPU-only (brak GPU)

| Ograniczenie | Wartość |
|-------------|---------|
| VRAM | 0 GB (RAM systemowy) |
| RAM | 8–16 GB |
| Główny problem | Brak akceleracji tensorowej |

**Strategia**: minimalizacja rozmiaru modelu i sekwencji, mały batch.

| Parametr | Zalecana wartość | Uzasadnienie |
|----------|:----------------:|-------------|
| `sequence_length` | 20–30 | Krótsze sekwencje = mniej obliczeń LSTM |
| `prediction_horizon` | 3–5 | Krótki horyzont = mniej kroków autoregresji |
| `encoder_hidden_dim` | 32–64 | Mała pojemność = szybciej |
| `encoder_num_layers` | 1 | Płytki enkoder |
| `latent_dim` | 16–32 | Mała przestrzeń latentna |
| `predictor_hidden_dim` | 32–64 | Mały predyktor |
| `predictor_num_layers` | 1 | Płytki predyktor |
| `batch_size` | 8–16 | Mały batch = mniej pamięci |
| `use_amp` | `false` | AMP nie działa na CPU |
| `num_workers` | 0–1 | 0 = uniknięcie problemów z multiprocessingiem |
| `features` | `[Close]` tylko | Minimalna liczba cech |

**Szacowany rozmiar modelu**: ~15–30K parametrów. Czas epoki (daily, 10 lat): ~5–15s.

### 2.2 GPU Low VRAM (4–8 GB)

| GPU | VRAM | Przykłady |
|-----|:----:|-----------|
| Niska półka | 4–8 GB | GTX 1650, RTX 3050, RTX 3060 (6GB), GTX 1080 |

**Strategia**: kompromis — umiarkowany model, AMP obowiązkowo.

| Parametr | Zalecana wartość | Uzasadnienie |
|----------|:----------------:|-------------|
| `sequence_length` | 30–60 | Standardowy kontekst |
| `prediction_horizon` | 5–10 | Umiarkowany horyzont |
| `encoder_hidden_dim` | 64–128 | Standard |
| `encoder_num_layers` | 1–2 | |
| `latent_dim` | 32–64 | |
| `predictor_hidden_dim` | 64–128 | |
| `predictor_num_layers` | 1–2 | |
| `batch_size` | 16–32 | 32 zmieści się z AMP |
| `use_amp` | `true` | **KLUCZOWE** — oszczędza ~40% VRAM |
| `pin_memory` | `true` | Przyspiesza transfer CPU→GPU |

**Szacowany rozmiar modelu**: ~200–400K parametrów. Czas epoki (daily, 10 lat): ~1–3s.

### 2.3 GPU Medium VRAM (12–16 GB)

| GPU | VRAM | Przykłady |
|-----|:----:|-----------|
| Średnia półka | 12–16 GB | RTX 3060 (12GB), RTX 4070, RTX 3080, RTX 4080 |

**Strategia**: pełna wersja domyślna, możliwość zwiększenia latent_dim i batch_size.

| Parametr | Zalecana wartość | Uzasadnienie |
|----------|:----------------:|-------------|
| `sequence_length` | 60–90 | Dłuższy kontekst |
| `prediction_horizon` | 5–15 | |
| `encoder_hidden_dim` | 128–256 | Większa pojemność |
| `encoder_num_layers` | 2–3 | |
| `latent_dim` | 64–128 | Większa przestrzeń latentna |
| `predictor_hidden_dim` | 128–256 | |
| `predictor_num_layers` | 2–3 | |
| `batch_size` | 32–64 | Większy batch = stabilniejsze gradienty |
| `use_amp` | `true` | Wciąż zalecane |

**Szacowany rozmiar modelu**: ~400K–1.5M parametrów. Czas epoki: ~0.5–2s.

### 2.4 GPU High VRAM (24+ GB)

| GPU | VRAM | Przykłady |
|-----|:----:|-----------|
| Wysoka półka | 24–80 GB | RTX 4090 (24GB), A5000, A100 (40/80GB), H100 |

**Strategia**: maksymalna jakość — duży model, długie sekwencje, wiele tickerów.

| Parametr | Zalecana wartość | Uzasadnienie |
|----------|:----------------:|-------------|
| `sequence_length` | 90–180 | Bardzo długi kontekst (6 miesięcy daily) |
| `prediction_horizon` | 10–30 | Długi horyzont predykcji |
| `encoder_hidden_dim` | 256–512 | |
| `encoder_num_layers` | 3–4 | |
| `latent_dim` | 128–256 | |
| `predictor_hidden_dim` | 256–512 | |
| `predictor_num_layers` | 3–4 | |
| `batch_size` | 64–256 | Duży batch = stabilne gradienty |
| `use_amp` | `true` | |

**Szacowany rozmiar modelu**: ~1.5–5M parametrów. Czas epoki: ~0.3–1s.

> **Uwaga**: Dla danych o wysokiej rozdzielczości patrz sekcja [3.4](#34-dane-o-wysokiej-rozdzielczości-1s--100ms) — tam VRAM rośnie drastycznie.

### 2.5 GPU RTX 50-series (Blackwell, GDDR7)

| GPU | VRAM | CUDA Cores | Tensor Cores | Bandwidth | Architektura |
|-----|:----:|:----------:|:------------:|:---------:|:------------:|
| RTX 5060 | 8 GB GDDR7 | ~4 608 | 5. gen (FP4/FP6/FP8) | ~448 GB/s | Blackwell |
| RTX 5070 | 12 GB GDDR7 | ~6 400 | 5. gen (FP4/FP6/FP8) | ~672 GB/s | Blackwell |
| RTX 5070 Ti | 16 GB GDDR7 | ~8 960 | 5. gen (FP4/FP6/FP8) | ~896 GB/s | Blackwell |
| RTX 5080 | 16 GB GDDR7 | 10 752 | 5. gen (FP4/FP6/FP8) | ~960 GB/s | Blackwell |
| RTX 5090 | 32 GB GDDR7 | 21 760 | 5. gen (FP4/FP6/FP8) | ~1.8 TB/s | Blackwell |

**Kluczowe cechy Blackwell dla R-JEPA**:

- **5. generacja Tensor Cores** — obsługa FP4, FP6, FP8 (oprócz standardowych FP16/BF16)
- **FP8 Transformer Engine** — dedykowane jednostki dla Transformerów, redukcja VRAM o ~50% względem FP16
- **GDDR7** — przepustowość pamięci wyższa o ~60% vs GDDR6X (RTX 4090: 1.0 TB/s, RTX 5090: 1.8 TB/s)
- **PCIe 5.0** — szybszy transfer danych CPU↔GPU (przydatne przy dużych datasetach intraday)

**Strategia**: maksymalnie wykorzystać FP8/FP6 oraz Transformer Encoder (zamiast LSTM) dla danych wysokiej rozdzielczości.

#### RTX 5090 (32 GB) — flagowa konfiguracja

| Parametr | Daily | Intraday 1min | High-freq (1s) |
|----------|:-----:|:--------------:|:--------------:|
| `sequence_length` | 180 | 240 | 1200 (z Conv1D) |
| `prediction_horizon` | 30 | 60 | 60 |
| `encoder_hidden_dim` | 512 | 512 | 384 |
| `encoder_num_layers` | 4 | 4 | — (Transformer) |
| `use_transformer` | `false` | `false` | **`true`** |
| `nhead` | — | — | 8 |
| `latent_dim` | 256 | 256 | 192 |
| `predictor_hidden_dim` | 512 | 512 | 384 |
| `batch_size` | 256 | 128 | 64 |
| `use_amp` | `true` (FP16) | `true` (FP16) | **`true` (FP8)** |
| `gradient_accumulation` | 1 | 2 | 4 |
| Czas epoki (est.) | ~0.2s | ~2s | ~15s |

#### RTX 5080 (16 GB) — wysoka średnia półka

| Parametr | Daily | Intraday 1min | High-freq (1s) |
|----------|:-----:|:--------------:|:--------------:|
| `sequence_length` | 120 | 180 | 600 (z Conv1D) |
| `prediction_horizon` | 20 | 30 | 30 |
| `encoder_hidden_dim` | 384 | 384 | 256 |
| `encoder_num_layers` | 3 | 3 | — (Transformer) |
| `use_transformer` | `false` | `false` | **`true`** |
| `nhead` | — | — | 8 |
| `latent_dim` | 192 | 128 | 128 |
| `batch_size` | 128 | 64 | 32 |
| `use_amp` | `true` (FP16) | `true` (FP16) | `true` (FP8) |
| `gradient_accumulation` | 1 | 2 | 4 |

#### RTX 5070 Ti / 5070 (12–16 GB) — średnia półka

| Parametr | Daily | Intraday 1min |
|----------|:-----:|:--------------:|
| `sequence_length` | 90 | 120 |
| `prediction_horizon` | 15 | 20 |
| `encoder_hidden_dim` | 256 | 256 |
| `encoder_num_layers` | 3 | 3 |
| `latent_dim` | 128 | 128 |
| `predictor_hidden_dim` | 256 | 256 |
| `batch_size` | 64–128 | 32–64 |
| `use_amp` | `true` (FP16) | `true` (FP16) |
| `gradient_accumulation` | 1 | 2 |

#### RTX 5060 (8 GB) — budżetowa Blackwell

| Parametr | Daily | Intraday 1min |
|----------|:-----:|:--------------:|
| `sequence_length` | 60 | 60 |
| `prediction_horizon` | 10 | 10 |
| `encoder_hidden_dim` | 128 | 128 |
| `encoder_num_layers` | 2 | 2 |
| `latent_dim` | 64 | 64 |
| `batch_size` | 32 | 16 |
| `use_amp` | `true` (FP16) | `true` (FP16) |
| `gradient_accumulation` | 1 | 2 |

**Przewaga Blackwell nad poprzednimi generacjami**:

| Cecha | RTX 4090 (Ada) | RTX 5090 (Blackwell) | Zysk |
|-------|:--------------:|:--------------------:|:----:|
| VRAM | 24 GB GDDR6X | 32 GB GDDR7 | +33% |
| Przepustowość | 1.0 TB/s | 1.8 TB/s | +80% |
| Tensor Cores | 4. gen (FP8) | 5. gen (FP4/FP6/FP8) | FP6 nowość |
| FP8 wydajność (TE) | ❌ | ✅ Transformer Engine | ~2x szybciej |
| PCIe | 4.0 | 5.0 | +100% transfer |

> **Wniosek**: RTX 5090 z FP8 Transformer Engine to najlepszy wybór dla danych wysokiej rozdzielczości. RTX 5060/5070 z GDDR7 oferują lepszy stosunek VRAM/ceny dzięki pamięci GDDR7.

---

## 3. Profile danych

### 3.1 Dane daily (domyślne)

| Charakterystyka | Wartość |
|----------------|---------|
| Liczba próbek | ~250/rok/ticker (sesje giełdowe) |
| 10 lat danych | ~2,500 próbek/ticker |
| Szum | Niski (ceny zamknięcia są stabilne) |
| Sezonowość | Dzienno-tygodniowo-miesięczna |

**Rekomendowana konfiguracja**:

```yaml
data:
  sequence_length: 60            # 3 miesiące sesji
  prediction_horizon: 5          # 1 tydzień
  batch_size: 32-64
  features: ["Open", "High", "Low", "Close", "Volume"]  # pełny zestaw

model:
  r_jepa:
    encoder_hidden_dim: 128
    encoder_num_layers: 2
    latent_dim: 64
    predictor_hidden_dim: 128
    predictor_num_layers: 2
```

**Liczba próbek**: 10 lat × ~250 = 2,500. Batch=64 → ~39 iteracji na epokę. Czas epoki: ~1s (GPU).

### 3.2 Dane intraday — 1 godzina

| Charakterystyka | Wartość |
|----------------|---------|
| Liczba próbek | ~6.5/dzień (rynek US: 9:30–16:00) |
| 1 rok danych | ~1,600 próbek |
| Szum | Średni (wahania wewnątrzdzienne) |
| Wzorce | Wewnątrzdzienna sezonowość (otwarcie, zamknięcie) |

**Rekomendowane zmiany względem daily**:

```yaml
data:
  sequence_length: 40            # ~1 tydzień godzinowy
  prediction_horizon: 6-12       # 1-2 dni
  batch_size: 32

model:
  r_jepa:
    encoder_hidden_dim: 128      # bez zmian
    encoder_num_layers: 2-3      # głębszy — więcej zmienności
    latent_dim: 64-96
    predictor_dropout: 0.2       # więcej regularyzacji (większy szum)
```

### 3.3 Dane intraday — 1 minuta

| Charakterystyka | Wartość |
|----------------|---------|
| Liczba próbek | ~390/dzień (6.5h × 60) |
| 1 miesiąc danych | ~8,200 próbek |
| Szum | Wysoki (szum mikrostruktury, spread bid-ask) |
| Wzorce | Krótkoterminowe (mean reversion, momentum) |

**Rekomendowane zmiany**:

```yaml
data:
  sequence_length: 120           # 2 godziny minutowych danych
  prediction_horizon: 30         # 30 minut w przyszłość
  batch_size: 16-32              # mniejszy batch przez VRAM

model:
  r_jepa:
    encoder_hidden_dim: 128-256
    encoder_num_layers: 3        # głębszy enkoder dla wysokiej zmienności
    latent_dim: 96-128
    predictor_hidden_dim: 128-256
    predictor_num_layers: 2-3
    predictor_dropout: 0.2-0.3   # silniejsza regularyzacja
    momentum_encoder_tau: 0.998  # wolniejsza EMA (więcej szumu w danych)
```

### 3.4 Dane o wysokiej rozdzielczości (1s / 100ms)

> Szczegółowa analiza w [`doc/implementacja.md`](implementacja.md:858) (§11.3.6).

| Rozdzielczość | Próbek/dzień | sequence_length | VRAM (batch=32) | Sugerowany GPU |
|:---:|:---:|:---:|:---:|:---:|
| **1 minuta** | ~390 | 120 | ~1.2 GB | RTX 3060 |
| **1 sekunda** | ~23,400 | 600 | ~4.8 GB | RTX 4090 |
| **100 ms** | ~234,000 | 6000 | ~24 GB | A100 |

**Kluczowe zmiany architektoniczne dla 1s / 100ms** (wymagają modyfikacji kodu):

```python
# W models/encoder.py — zastąpienie/rozszerzenie o Conv1D downsample
self.conv_downsample = nn.Sequential(
    nn.Conv1d(input_dim, hidden_dim, kernel_size=32, stride=4, padding=16),
    nn.BatchNorm1d(hidden_dim),
    nn.GELU(),
    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=16, stride=2, padding=8),
    nn.BatchNorm1d(hidden_dim),
    nn.GELU(),
    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=8, stride=1, padding=4),
)
# Wynik: T=600 → 150 → 75 → 75 tokenów
```

**Konfiguracja dla 1s (RTX 4090, 24GB)**:

```yaml
data:
  sequence_length: 600           # 10 minut
  prediction_horizon: 60         # 1 minuta
  batch_size: 32
  features: ["Open", "High", "Low", "Close", "Volume",
             "Spread", "Volume_Imbalance", "Trade_Intensity"]  # rozszerzone

model:
  r_jepa:
    encoder_hidden_dim: 256      # Conv1D + Transformer
    encoder_num_layers: 4        # Transformer(4 warstwy, nhead=8)
    latent_dim: 128
    predictor_hidden_dim: 256
    predictor_num_layers: 2
    predictor_dropout: 0.3       # silna regularyzacja dla szumnych danych
    momentum_encoder_tau: 0.999  # bardzo wolna EMA

training:
  use_amp: true                  # OBOWIĄZKOWE
  gradient_clip_val: 0.5         # niższy clipping dla stabilności
  num_epochs: 50
  patience: 15                   # więcej cierpliwości
```

### 3.5 Wiele tickerów jednocześnie

| Tickerów | Dane (batch) | Rozmiar modelu | Strategia |
|:--------:|:------------:|:--------------:|-----------|
| 1–3 | Konkatenacja w osi 0 | Jeden model | Domyślna |
| 5–10 | Konkatenacja | Jeden model | Większy batch |
| 10+ | Osobne modele | Jeden/ticker | Paralelizacja |

**Konfiguracja dla 10 tickerów**:

```yaml
data:
  tickers: ["AAPL", "MSFT", "GOOGL", "AMZN", "META",
            "TSLA", "NVDA", "JPM", "V", "JNJ"]
  batch_size: 128                # Więcej danych = większy batch

model:
  r_jepa:
    encoder_hidden_dim: 256      # Większa pojemność dla różnych wzorców
    latent_dim: 128
```

**Uwaga**: Dane tickerów są konkatenowane w osi czasu w [`data/stock_data.py`](../data/stock_data.py:55) (`np.concatenate(all_data, axis=0)`). Każda próbka w batchu może pochodzić z innego tickera.

---

## 4. Szczegółowy opis parametrów

### 4.1 Parametry danych

| Parametr | Typ | Domyślny | Zakres | Opis |
|----------|:---:|:--------:|:------:|------|
| `tickers` | `list[str]` | `[AAPL, MSFT, GOOGL]` | dowolne tickery | Lista tickerów Yahoo Finance |
| `start_date` | `str` | `2015-01-01` | YYYY-MM-DD | Początek zakresu danych |
| `end_date` | `str` | `2024-12-31` | YYYY-MM-DD | Koniec zakresu danych |
| `sequence_length` | `int` | 60 | 10–600+ | Liczba kroków czasowych kontekstu (past) |
| `prediction_horizon` | `int` | 5 | 1–60+ | Liczba kroków do przewidzenia (future) |
| `train_split` | `float` | 0.8 | 0.5–0.9 | Frakcja danych treningowych |
| `val_split` | `float` | 0.1 | 0.05–0.2 | Frakcja danych walidacyjnych |
| `batch_size` | `int` | 64 | 1–512 | Rozmiar batcha |
| `features` | `list[str]` | OHLCV | zależne | Które cechy użyć |
| `normalize` | `bool` | `true` | — | Czy normalizować StandardScalerem |

**Wpływ `sequence_length`**:
- Za krótki (< 20): model nie widzi wystarczającego kontekstu (trendów, sezonowości)
- Za długi (> 180 dla daily): malejące korzyści, rosnące koszty obliczeniowe
- Dla daily: 60 = ~3 miesiące sesji — dobry balans

**Wpływ `prediction_horizon`**:
- Za krótki (1–3): trywialne (predykcja blisko teraźniejszości)
- Za długi (> 20 dla daily): bardzo trudne, wysoka niepewność
- Dla daily: 5 = 1 tydzień — rozsądny horyzont

### 4.2 Parametry enkodera

Zdefiniowane w [`models/encoder.py`](../models/encoder.py:20) (`TimeSeriesEncoder.__init__`).

| Parametr | Typ | Domyślny | Zakres | Opis |
|----------|:---:|:--------:|:------:|------|
| `encoder_input_dim` | `int` | 5 (ilość cech) | — | Automatycznie z danych |
| `encoder_hidden_dim` | `int` | 128 | 32–1024 | Wymiar ukryty Conv1D i LSTM |
| `encoder_num_layers` | `int` | 2 | 1–6 | Liczba warstw LSTM |
| `latent_dim` | `int` | 64 | 16–512 | Wymiar przestrzeni latentnej `z` |

**Architektura enkodera** (zob. [`models/encoder.py`](../models/encoder.py:30)):
1. Conv1D(k=3, padding=1) → BatchNorm → GELU → Conv1D → BatchNorm → GELU
2. BiLSTM (liczba warstw = `encoder_num_layers`)
3. Projection MLP → [`latent_dim`]

**Wpływ `encoder_hidden_dim`**:
- Parametry LSTM: `4 × (hidden_dim × input_size + hidden_dim² + hidden_dim) × num_layers`
- Dla `hidden_dim=128`: ~264K parametrów w LSTM (przy input_dim=128, 2 warstwy)
- Dla `hidden_dim=256`: ~1M parametrów (~4x więcej)
- **Wybór**: 64 dla CPU, 128 dla GPU średnia, 256+ dla GPU wysoka

**Wpływ `encoder_num_layers`**:
- Każda dodatkowa warstwa LSTM dodaje ~`4 × (hidden_dim² + hidden_dim)` parametrów
- Dla `hidden_dim=128`: ~66K parametrów/warstwę
- Głębsze warstwy → lepsze hierarchiczne reprezentacje, ale wolniejsze i więcej VRAM

**Wpływ `latent_dim`**:
- Kluczowy parametr jakości — determinuje pojemność przestrzeni reprezentacji
- Zbyt mały (< 16): bottleneck — model nie może wyrazić złożonych wzorców
- Zbyt duży (> 256): nadmierna pojemność, ryzyko collapse'u, więcej VRAM
- **Rekomendacja**: 32–64 dla daily, 64–128 dla intraday

### 4.3 Parametry predyktora

Zdefiniowane w [`models/predictor.py`](../models/predictor.py:21) (`RecurrentPredictor.__init__`).

| Parametr | Typ | Domyślny | Zakres | Opis |
|----------|:---:|:--------:|:------:|------|
| `predictor_hidden_dim` | `int` | 128 | 32–1024 | Wymiar ukryty GRU |
| `predictor_num_layers` | `int` | 2 | 1–6 | Liczba warstw GRU |
| `predictor_dropout` | `float` | 0.1 | 0–0.5 | Dropout w predyktorze |

**Architektura predyktora**:
1. Context Aggregator: Linear(`latent_dim`→`hidden_dim`) → LayerNorm → GELU
2. GRU (`latent_dim`→`hidden_dim`, `num_layers` warstw)
3. Output Projection: Linear(`hidden_dim`→`hidden_dim`) → LayerNorm → GELU → Dropout → Linear(`hidden_dim`→`latent_dim`)
4. Autoregresywna pętla: `pred_horizon` kroków, każdy GRU krok używa poprzedniej predykcji

**Wpływ `predictor_dropout`**:
- Dla daily (mało szumu): 0.05–0.1
- Dla intraday (więcej szumu): 0.2–0.3
- Zbyt wysoki (> 0.4): underfitting, model nie może się uczyć

### 4.4 Parametry dekodera

Zdefiniowane w [`models/decoder.py`](../models/decoder.py:18) (`StockDecoder.__init__`).

| Parametr | Typ | Domyślny | Zakres | Opis |
|----------|:---:|:--------:|:------:|------|
| `decoder_hidden_dim` | `int` | 64 | 32–256 | Wymiar warstw ukrytych MLP |
| `decoder_output_dim` | `int` | auto | = liczba cech | Automatycznie (tyle samo co wejście) |

**Architektura dekodera**: MLP 3-warstwowy:
`latent_dim → hidden_dim → LayerNorm → GELU → hidden_dim → LayerNorm → GELU → output_dim`

Dekoder jest używany do:
1. Obliczania straty rekonstrukcyjnej podczas treningu
2. Dekodowania predykcji podczas inferencji

**Uwaga**: Dekoder ma relatywnie mały wpływ na VRAM (~1–5% parametrów). Można go swobodnie zwiększać.

### 4.5 Parametry JEPA

| Parametr | Typ | Domyślny | Zakres | Opis |
|----------|:---:|:--------:|:------:|------|
| `momentum_encoder_tau` | `float` | 0.996 | 0.99–0.9999 | Współczynnik EMA enkodera momentum |
| `predictor_epsilon` | `float` | 0.001 | 1e-6–0.01 | Stabilność numeryczna straty |
| `variance_weight` | `float` | 0.5 | 0.1–2.0 | Waga regularyzacji wariancji |
| `covariance_weight` | `float` | 0.1 | 0.01–0.5 | Waga regularyzacji kowariancji |

**Wpływ `momentum_encoder_tau`**:
- `tau` bliższe 1.0 → wolniejsza aktualizacja → stabilniejsze targety
- Dla daily (stabilne dane): `tau=0.996`
- Dla intraday (szumne dane): `tau=0.998–0.999` (wolniejsza EMA filtruje szum)
- Za niskie `tau` (< 0.99): cele zmieniają się zbyt szybko, niestabilny trening
- Za wysokie `tau` (> 0.9999): cele zmieniają się zbyt wolno, model nie nadąża

**Wpływ `variance_weight` i `covariance_weight`**:
- Regularyzacja VCReg (Variance-Covariance Regularization) zapobiega collapse'owi
- `variance_weight=0.5` — domyślna, dobra dla większości przypadków
- Dla małych batchy (< 16): zwiększyć `variance_weight` do 1.0 (większe ryzyko collapse'u)
- Dla dużych latent_dim (> 128): zwiększyć `covariance_weight` do 0.2 (więcej wymiarów do decorrelacji)

### 4.6 Parametry treningu

Zdefiniowane w [`training/trainer.py`](../training/trainer.py:39) i [`config/config.yaml`](../config/config.yaml:37).

| Parametr | Typ | Domyślny | Zakres | Opis |
|----------|:---:|:--------:|:------:|------|
| `learning_rate` | `float` | 0.001 | 1e-5–1e-2 | Początkowy learning rate |
| `weight_decay` | `float` | 0.0001 | 0–0.01 | Regularyzacja L2 (AdamW) |
| `num_epochs` | `int` | 100 | 20–500 | Maksymalna liczba epok |
| `patience` | `int` | 10 | 5–30 | Early stopping patience |
| `gradient_clip_val` | `float` | 1.0 | 0.1–5.0 | Maksymalna norma gradientu |
| `use_amp` | `bool` | `true` | — | Automatic Mixed Precision |
| `log_interval` | `int` | 10 | 1–50 | Co ile epok logować |
| `save_dir` | `str` | ./checkpoints | — | Katalog na checkpointy |

**Wpływ `learning_rate`**:
- Dla małych modeli (< 100K params): `lr=0.001–0.005`
- Dla dużych modeli (> 500K params): `lr=0.0005–0.001`
- Z schedulerem CosineAnnealingLR (od `lr` do `eta_min=1e-6`)

**Wpływ `gradient_clip_val`**:
- Dla daily (stabilne gradienty): `1.0`
- Dla intraday (niestabilne gradienty): `0.5–1.0`
- Zbyt niski (< 0.1): gradienty obcięte zbyt mocno, model nie uczy się
- Zbyt wysoki (> 5.0): brak ochrony przed eksplodującymi gradientami

**Wpływ `use_amp`** (zob. [`training/trainer.py`](../training/trainer.py:84)):
```python
scaler = GradScaler(enabled=self.use_amp)  # trainer.py:85
# Automatyczne fp16/fp32 mieszanie
with autocast(enabled=self.use_amp):        # trainer.py:182
    output = model(context, target)
    loss = criterion(...)
scaler.scale(loss).backward()               # trainer.py:193
```
- Redukcja VRAM: ~40%
- Przyspieszenie: ~1.5–2.5x (zależne od GPU — największe na RTX 30xx/40xx z Tensor Cores)
- Wpływ na jakość: minimalny (loss w fp32, aktywacje w fp16 gdzie bezpiecznie)
- **Zalecane**: `true` dla wszystkich GPU z supportem (RTX 20xx+, A100, H100)

---

## 5. Macierz zależności sprzęt–parametry

| Scenariusz | VRAM | batch_size | seq_len | latent_dim | hidden_dim | AMP | Czas epoki |
|-----------|:----:|:----------:|:-------:|:----------:|:----------:|:---:|:----------:|
| CPU-only | RAM 8GB | 8 | 20 | 16 | 32 | ✗ | 5–15s |
| GTX 1650 (4GB) | 4 GB | 16 | 30 | 32 | 64 | ✓ | 3–8s |
| RTX 3060 (6GB) | 6 GB | 16 | 60 | 64 | 128 | ✓ | 1–3s |
| RTX 3060 (12GB) | 12 GB | 32 | 60 | 64 | 128 | ✓ | 1–2s |
| RTX 4070 (12GB) | 12 GB | 64 | 60 | 64 | 128 | ✓ | 0.5–1s |
| RTX 4090 (24GB) | 24 GB | 128 | 90 | 128 | 256 | ✓ | 0.3–0.8s |
| RTX 5060 (8GB, GDDR7) | 8 GB | 32 | 60 | 64 | 128 | ✓ FP8 | 0.5–1s |
| RTX 5070 (12GB, GDDR7) | 12 GB | 64 | 90 | 128 | 256 | ✓ FP8 | 0.3–0.8s |
| RTX 5070 Ti (16GB, GDDR7) | 16 GB | 128 | 90 | 128 | 256 | ✓ FP8 | 0.3–0.6s |
| RTX 5080 (16GB, GDDR7) | 16 GB | 128 | 120 | 192 | 384 | ✓ FP8 | 0.2–0.5s |
| RTX 5090 (32GB, GDDR7) | 32 GB | 256 | 180 | 256 | 512 | ✓ FP8 | 0.15–0.3s |
| A100 (40GB) | 40 GB | 256 | 120 | 256 | 512 | ✓ | 0.2–0.5s |
| **1s daily** (RTX 4090) | 24 GB | 32 | 600 | 128 | 256 | ✓ | ~32s |
| **1s daily** (A100) | 40 GB | 64 | 600 | 128 | 256 | ✓ | ~18s |

> **Uwaga**: Czasy dla danych daily (~2,500 próbek). Dla intraday czasy proporcjonalnie dłuższe.

---

## 6. Drzewo decyzyjne — jak dobrać konfigurację

```
Czy masz GPU?
├── NIE → CPU Profile (#2.1)
│   └── sequence_length=20, latent_dim=16, batch=8
│
└── TAK → Jaki VRAM?
    ├── 4–8 GB → Low VRAM Profile (#2.2)
    │   ├── Czy dane są daily?
    │   │   ├── TAK → batch=16, latent_dim=32-64, hidden_dim=64-128
    │   │   └── NIE (intraday 1h) → batch=16, seq_len=40, latent_dim=32
    │   └── Włącz AMP: use_amp=true (obowiązkowo)
    │
    ├── 12–16 GB → Medium VRAM Profile (#2.3)
    │   ├── Jakie dane?
    │   │   ├── daily → Standard: seq_len=60, latent_dim=64, batch=32-64
    │   │   ├── intraday 1h → seq_len=40, latent_dim=96, batch=32
    │   │   └── intraday 1min → seq_len=120, latent_dim=96, batch=16-32
    │   └── AMP zalecany
    │
    ├── 16–24 GB → RTX 50-series Blackwell? (#2.5)
    │   ├── NIE → Medium VRAM Profile (RTX 3080/4080)
    │   │   └── Jak §2.3, AMP FP16
    │   │
    │   └── TAK → Blackwell GPU
    │       ├── 8 GB (RTX 5060) → §2.5.4
    │       │   └── seq_len=60, latent_dim=64, batch=32, AMP FP8
    │       ├── 12 GB (RTX 5070) → §2.5.3
    │       │   └── seq_len=90, latent_dim=128, batch=64, AMP FP8
    │       ├── 16 GB (5070 Ti / 5080) → §2.5.2
    │       │   ├── daily → seq_len=120, latent_dim=192, batch=128, AMP FP8
    │       │   └── 1s → seq_len=600, Transformer, nhead=8, batch=32, FP8
    │       └── 32 GB (RTX 5090) → §2.5.1
    │           ├── daily → seq_len=180, latent_dim=256, batch=256, AMP FP8
    │           ├── 1min → seq_len=240, latent_dim=256, batch=128, AMP FP8
    │           └── 1s → seq_len=1200, Transformer, nhead=8, batch=64, FP8
    │
    └── 24+ GB → High VRAM Profile (#2.4)
        ├── RTX 5090? → patrz §2.5.1 (Blackwell + FP8)
        ├── Jakie dane?
        │   ├── daily → seq_len=90-180, latent_dim=128-256, batch=64-256
        │   ├── intraday 1min → seq_len=120, latent_dim=128, batch=64
        │   └── 1s → seq_len=600, latent_dim=128, batch=32-64 (Conv1D mods)
        └── AMP włączony

Następnie:

Czy używasz wielu tickerów?
├── 1–3 → batch jak wyżej
├── 5–10 → batch ×2–3 (więcej danych na epokę)
└── 10+ → batch ×4+, hidden_dim = 128-256 (większa pojemność)

Czy dane są szumne? (intraday, kryptowaluty)
├── TAK → zwiększ dropout (0.2-0.3), zmniejsz gradient_clip (0.5),
│         zwiększ momentum_tau (0.998-0.999)
└── NIE → standardowe parametry
```

---

## 7. Optymalizacja wydajności

### 7.1 Automatic Mixed Precision (AMP)

Zastosowany w [`training/trainer.py`](../training/trainer.py:84) — włączany przez `use_amp: true`.

```python
# Automatyczne przełączanie między fp16 i fp32
scaler = GradScaler(enabled=True)

with autocast(enabled=True):
    output = model(context, target)
    loss = criterion(output)

scaler.scale(loss).backward()     # Skaluje loss → backward w fp16
scaler.step(optimizer)            # Odskaluje gradienty → update wag w fp32
scaler.update()                   # Aktualizuje skalę dla następnego batcha
```

| Aspekt | Wartość |
|--------|---------|
| Redukcja VRAM | ~35–45% |
| Przyspieszenie | ~1.5–2.5x |
| Wymagany GPU | RTX 20xx+ (Tensor Cores) lub A100/H100 |
| **FP8 (RTX 50-series Blackwell)** | Dodatkowa redukcja VRAM ~20% vs FP16, ~2x faster matmul |
| Wpływ na jakość | < 0.1% różnica w loss dla FP16; < 0.5% dla FP8 (zalecany tylko dla Transformer) |
| **Zalecenie** | **FP16 — zawsze; FP8 — RTX 50-series + Transformer + dane HF** |

### 7.2 Gradient Accumulation

Przydatne gdy VRAM nie pozwala na duży batch. Efektywny batch = `batch_size × accumulation_steps`.

```python
accumulation_steps = 4  # efektywny batch = 16 × 4 = 64
optimizer.zero_grad()

for i, (context, target) in enumerate(train_loader):
    with autocast(enabled=True):
        output = model(context, target)
        loss = criterion(output) / accumulation_steps  # skalowanie
    
    scaler.scale(loss).backward()
    
    if (i + 1) % accumulation_steps == 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
```

| accumulation_steps | VRAM (batch=16) | Efektywny batch | Czas epoki |
|:---:|:---:|:---:|:---:|
| 1 | ~1.5 GB | 16 | 1× |
| 2 | ~1.5 GB | 32 | 1.05× |
| 4 | ~1.5 GB | 64 | 1.1× |
| 8 | ~1.5 GB | 128 | 1.2× |

**Kiedy użyć**: gdy batch_size jest ograniczony przez VRAM, a potrzebujesz większego efektywnego batcha dla stabilnych gradientów.

### 7.3 Gradient Checkpointing

Redukuje VRAM kosztem czasu — nie zapamiętuje wszystkich aktywacji, tylko przelicza je w backward passie.

```python
# Przykład: opakowanie warstwy w checkpoint
from torch.utils.checkpoint import checkpoint

def forward(self, x):
    # Zamiast:
    # conv_out = self.conv_layers(x)
    
    # Użyj:
    conv_out = checkpoint(self.conv_layers, x, use_reentrant=False)
    
    # ... reszta forward pass
```

| Aspekt | Wartość |
|--------|---------|
| Redukcja VRAM | ~40–60% |
| Spowolnienie | ~15–25% |
| Implementacja | Wymaga modyfikacji kodu modelu |
| **Kiedy użyć** | Gdy VRAM jest wąskim gardłem, a czas jest akceptowalny |

### 7.4 Batch Size tuning

`batch_size` to najprostszy parametr do kontroli VRAM. Zależność VRAM od batch_size jest **liniowa**.

```python
# Szacowanie VRAM dla różnych batch_size (daily, seq_len=60, latent_dim=64)
batch_sizes = [8, 16, 32, 64, 128, 256]
for bs in batch_sizes:
    vram_gb = 0.3 + bs * 0.012  # przybliżenie
    print(f"batch={bs}: ~{vram_gb:.1f} GB VRAM")
```

| batch_size | Szac. VRAM | Zalecany GPU |
|:----------:|:----------:|--------------|
| 8 | ~0.4 GB | CPU / dowolny |
| 16 | ~0.5 GB | GTX 1650 (4GB) |
| 32 | ~0.7 GB | RTX 3060 (6GB) |
| 64 | ~1.1 GB | RTX 3060 (12GB) |
| 128 | ~1.8 GB | RTX 4070 (12GB) |
| 256 | ~3.3 GB | RTX 4090 (24GB) |

> **Dla 1s danych** (seq_len=600): batch=32 → ~4.8 GB (zob. [`implementacja.md`](implementacja.md:872))

### 7.5 num_workers i pin_memory

Parametry [`create_dataloaders`](../data/stock_data.py:185) wpływające na szybkość ładowania danych.

```python
loader_kwargs = dict(
    batch_size=batch_size,
    num_workers=num_workers,          # >= 2 dla GPU, 0 dla CPU
    pin_memory=torch.cuda.is_available(),  # True dla GPU
)
```

| Parametr | CPU | GPU |
|----------|:---:|:---:|
| `num_workers` | 0–1 | 2–8 |
| `pin_memory` | `false` | `true` |

**Zalecenia**:
- `num_workers=2` dla GPU średnia, `num_workers=4–8` dla GPU wysoka
- `pin_memory=true` przyspiesza transfer CPU→GPU (działa tylko z CUDA)
- Dla CPU: `num_workers=0` (uniknięcie multiprocessing overhead)

---

## 8. Przepisy — gotowe konfiguracje YAML

### 8.1 Lekka — CPU / laptop

```yaml
# config_lekka.yaml
data:
  tickers: ["AAPL"]
  start_date: "2020-01-01"
  end_date: "2024-12-31"
  sequence_length: 20
  prediction_horizon: 5
  train_split: 0.8
  val_split: 0.1
  batch_size: 8
  features: ["Close"]           # tylko jedna cecha
  normalize: true

model:
  r_jepa:
    encoder_hidden_dim: 32
    encoder_num_layers: 1
    latent_dim: 16
    predictor_hidden_dim: 32
    predictor_num_layers: 1
    predictor_dropout: 0.05
    decoder_hidden_dim: 16
    momentum_encoder_tau: 0.996

training:
  learning_rate: 0.001
  weight_decay: 0.0001
  num_epochs: 50
  patience: 10
  gradient_clip_val: 1.0
  use_amp: false                 # CPU nie wspiera AMP
  log_interval: 10
  save_dir: "./checkpoints"
```

**Uruchomienie**:
```bash
python run_training.py --config config/config_lekka.yaml --device cpu
```

### 8.2 Standard — RTX 3060/4070, daily

```yaml
# config_standard.yaml
data:
  tickers: ["AAPL", "MSFT", "GOOGL"]
  start_date: "2010-01-01"
  end_date: "2024-12-31"
  sequence_length: 60
  prediction_horizon: 5
  train_split: 0.8
  val_split: 0.1
  batch_size: 32
  features: ["Open", "High", "Low", "Close", "Volume"]
  normalize: true

model:
  r_jepa:
    encoder_hidden_dim: 128
    encoder_num_layers: 2
    latent_dim: 64
    predictor_hidden_dim: 128
    predictor_num_layers: 2
    predictor_dropout: 0.1
    decoder_hidden_dim: 64
    momentum_encoder_tau: 0.996

training:
  learning_rate: 0.001
  weight_decay: 0.0001
  num_epochs: 100
  patience: 10
  gradient_clip_val: 1.0
  use_amp: true
  log_interval: 10
  save_dir: "./checkpoints"
```

### 8.3 Szybka — RTX 4090, daily

```yaml
# config_szybka.yaml
data:
  tickers: ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
  start_date: "2010-01-01"
  end_date: "2024-12-31"
  sequence_length: 90              # dłuższy kontekst
  prediction_horizon: 10           # dłuższy horyzont
  train_split: 0.8
  val_split: 0.1
  batch_size: 128                  # duży batch
  features: ["Open", "High", "Low", "Close", "Volume"]
  normalize: true

model:
  r_jepa:
    encoder_hidden_dim: 256
    encoder_num_layers: 3
    latent_dim: 128
    predictor_hidden_dim: 256
    predictor_num_layers: 3
    predictor_dropout: 0.15
    decoder_hidden_dim: 128
    momentum_encoder_tau: 0.996

training:
  learning_rate: 0.0008            # niższy LR dla większego modelu
  weight_decay: 0.0001
  num_epochs: 150
  patience: 15
  gradient_clip_val: 1.0
  use_amp: true
  log_interval: 5
  save_dir: "./checkpoints"
```

### 8.4 Wielotickerowa — RTX 4090, 10+ tickerów

```yaml
# config_multiticker.yaml
data:
  tickers: ["AAPL", "MSFT", "GOOGL", "AMZN", "META",
            "TSLA", "NVDA", "JPM", "V", "JNJ"]
  start_date: "2015-01-01"
  end_date: "2024-12-31"
  sequence_length: 60
  prediction_horizon: 5
  train_split: 0.8
  val_split: 0.1
  batch_size: 256                  # dużo danych, duży batch
  features: ["Open", "High", "Low", "Close", "Volume"]
  normalize: true

model:
  r_jepa:
    encoder_hidden_dim: 256        # większa pojemność
    encoder_num_layers: 3
    latent_dim: 128
    predictor_hidden_dim: 256
    predictor_num_layers: 3
    predictor_dropout: 0.15
    decoder_hidden_dim: 128
    momentum_encoder_tau: 0.996

training:
  learning_rate: 0.001
  weight_decay: 0.0001
  num_epochs: 150
  patience: 20                    # więcej cierpliwości (więcej danych)
  gradient_clip_val: 1.0
  use_amp: true
  log_interval: 5
  save_dir: "./checkpoints"
```

### 8.5 High-frequency — A100, 1-minuta

```yaml
# config_hf.yaml
data:
  tickers: ["AAPL"]
  start_date: "2024-01-01"
  end_date: "2024-06-30"
  sequence_length: 120             # 2 godziny minutowych danych
  prediction_horizon: 30           # 30 minut
  train_split: 0.8
  val_split: 0.1
  batch_size: 64
  features: ["Open", "High", "Low", "Close", "Volume"]
  normalize: true

model:
  r_jepa:
    encoder_hidden_dim: 256
    encoder_num_layers: 3          # głębszy dla intraday zmienności
    latent_dim: 128
    predictor_hidden_dim: 256
    predictor_num_layers: 3
    predictor_dropout: 0.25        # silniejsza regularyzacja
    decoder_hidden_dim: 128
    momentum_encoder_tau: 0.998    # wolniejsza EMA

training:
  learning_rate: 0.0005            # niższy LR dla szumnych danych
  weight_decay: 0.0005             # więcej regularyzacji L2
  num_epochs: 100
  patience: 15
  gradient_clip_val: 0.5           # niższy clipping
  use_amp: true                    # OBOWIĄZKOWE
  log_interval: 10
  save_dir: "./checkpoints"
```

### 8.6 RTX 5090 — high-frequency, Transformer

Dedykowana konfiguracja dla RTX 5090 (32 GB GDDR7, Blackwell) z wykorzystaniem **FP8 Transformer Engine** i **Conv1D downsamplingu** dla danych 1-sekundowych. Wykorzystuje [`TransformerEncoder`](../models/encoder.py) zamiast LSTM dla lepszej obsługi długich sekwencji.

```yaml
# config_rtx5090_hf.yaml
data:
  tickers: ["AAPL"]
  start_date: "2024-01-01"
  end_date: "2024-06-30"
  sequence_length: 1200             # 20 minut danych 1s (z Conv1D downsample)
  prediction_horizon: 60            # 1 minuta
  train_split: 0.8
  val_split: 0.1
  batch_size: 64
  sampling_interval: "1s"           # dane wysokiej rozdzielczości
  use_high_freq_features: true      # Spread, Volume_Imbalance, itp.
  features: ["Open", "High", "Low", "Close", "Volume",
             "Spread", "Volume_Imbalance", "Trade_Intensity"]
  normalize: true

model:
  r_jepa:
    conv_downsample: true            # Conv1D: k=32→k=16→k=8 (600→75 tokenów)
    use_transformer: true            # TransformerEncoder zamiast LSTM
    nhead: 8                         # 8 głowic attentions
    encoder_hidden_dim: 384
    encoder_num_layers: 4            # 4 warstwy Transformer
    latent_dim: 192
    predictor_hidden_dim: 384
    predictor_num_layers: 3
    predictor_dropout: 0.2
    decoder_hidden_dim: 192
    momentum_encoder_tau: 0.998

training:
  learning_rate: 0.0003              # niższy LR dla Transformerów
  weight_decay: 0.0005
  num_epochs: 100
  patience: 15
  gradient_clip_val: 0.5
  use_amp: true                      # AMP w trybie FP8 (Blackwell Tensor Cores)
  gradient_accumulation_steps: 4     # efektywny batch = 64 × 4 = 256
  log_interval: 10
  save_dir: "./checkpoints"
```

**Uruchomienie**:
```bash
python run_training.py --config config/config_rtx5090_hf.yaml \
    --conv_downsample --use_transformer --nhead 8 \
    --sampling_interval 1s --high_freq_features
```

> ⚠️ **Wymagania**: PyTorch 2.4+ z obsługą FP8 (NVIDIA Blackwell). Bez FP8 konfiguracja działa w FP16 z nieco mniejszym batch_size (48 zamiast 64).

**Szacowany VRAM**: ~24–28 GB z FP8, ~30–32 GB z FP16. Czas epoki (6 miesięcy danych 1s): ~15–25s.

---

## 9. Monitorowanie i diagnostyka

### 9.1 Sprawdzanie wykorzystania VRAM

```python
import torch

def print_vram_usage():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"VRAM allocated: {allocated:.2f} GB")
        print(f"VRAM reserved:  {reserved:.2f} GB")
        print(f"VRAM total:     {torch.cuda.get_device_properties(0).total_memory / 1024**3:.0f} GB")
```

### 9.2 Monitorowanie podczas treningu

Dodaj do [`run_training.py`](../run_training.py) opcjonalne monitorowanie:

```python
# W pętli treningowej lub jako callback
if epoch % 10 == 0 and torch.cuda.is_available():
    print(f"  VRAM: {torch.cuda.memory_allocated() / 1024**3:.2f} GB / "
          f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.0f} GB")
```

### 9.3 Sygnały ostrzegawcze

| Objaw | Prawdopodobna przyczyna | Rozwiązanie |
|-------|------------------------|-------------|
| Loss = NaN | Eksplodujące gradienty | Zmniejsz `gradient_clip_val` lub `learning_rate` |
| Loss nie spada | Za mały model / za mało danych | Zwiększ `latent_dim` lub `hidden_dim` |
| Loss walidacyjny rośnie | Przeuczenie (overfitting) | Zwiększ `predictor_dropout`, `weight_decay` |
| VRAM OOM (Out of Memory) | Za duży batch / model | Zmniejsz `batch_size`, włącz AMP |
| Bardzo wolny trening (CPU) | Za duży model dla CPU | Użyj profilu CPU-only (#8.1) |
| Collapse (wszystkie wyjścia = const) | Za słaba regularyzacja VCReg | Zwiększ `variance_weight` lub zmniejsz `latent_dim` |

---

> **Zobacz także**:
> - [`implementacja.md`](implementacja.md) — szczegóły implementacji każdego komponentu
> - [`proces_uczenia.md`](proces_uczenia.md) — przebieg treningu krok po kroku
> - [`config/config.yaml`](../config/config.yaml) — domyślna konfiguracja
