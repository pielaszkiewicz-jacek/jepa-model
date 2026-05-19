# 🏗️ Architektura R-JEPA

> Diagramy architektury, przepływy danych i komponenty systemu.

---

## Spis treści

- [🏗️ Architektura R-JEPA](#️-architektura-r-jepa)
  - [Spis treści](#spis-treści)
  - [1. Architektura wysokiego poziomu](#1-architektura-wysokiego-poziomu)
    - [1.1 Diagram główny](#11-diagram-główny)
    - [1.2 Komponenty według warstw](#12-komponenty-według-warstw)
  - [2. Przepływ danych](#2-przepływ-danych)
    - [2.1 Przepływ treningowy](#21-przepływ-treningowy)
    - [2.2 Przepływ inferencyjny](#22-przepływ-inferencyjny)
  - [3. Komponenty systemu](#3-komponenty-systemu)
    - [3.1 StockDataLoader](#31-stockdataloader)
    - [3.2 Model R-JEPA](#32-model-r-jepa)
    - [3.3 Training](#33-training)
  - [4. Tryb treningowy vs inferencyjny](#4-tryb-treningowy-vs-inferencyjny)
  - [5. Diagram sekwencji — pełny cykl życia](#5-diagram-sekwencji--pełny-cykl-życia)
  - [6. Zależności między modułami](#6-zależności-między-modułami)

---

## 1. Architektura wysokiego poziomu

### 1.1 Diagram główny

```mermaid
flowchart TB
    subgraph Data["📊 Data Layer"]
        YF["yfinance<br/><small>Yahoo Finance API</small>"]
        ENG["Feature Engineering<br/><small>RSI, MA, Volatility, ...</small>"]
        SC["StandardScaler<br/><small>Normalizacja</small>"]
        DS["StockDataset<br/><small>Context-Target pairs</small>"]
    end

    subgraph Model["🧠 Model Layer"]
        OE["Online Encoder<br/><small>Conv1D → BiLSTM</small>"]
        ME["Momentum Encoder<br/><small>EMA copy</small>"]
        RP["Recurrent Predictor<br/><small>GRU</small>"]
        DEC["Decoder<br/><small>Latent → Price</small>"]
    end

    subgraph Training["⚙️ Training Layer"]
        LOSS["JEPA Loss<br/><small>MSE + Var + Cov</small>"]
        OPT["AdamW Optimizer"]
        SCH["CosineAnnealingLR"]
        CKPT["Checkpoint Manager"]
    end

    subgraph Inference["🔮 Inference Layer"]
        ENS["Ensemble Generator<br/><small>Noise injection</small>"]
        CI["Confidence Intervals<br/><small>95% CI</small>"]
        PLOT["Visualization<br/><small>Matplotlib</small>"]
    end

    YF --> ENG --> SC --> DS
    DS --> OE
    DS --> ME

    OE --> RP
    ME --> LOSS
    RP --> LOSS
    LOSS --> OPT --> OE
    OPT --> SCH

    RP --> DEC
    DEC --> ENS --> CI --> PLOT

    style Data fill:#e3f2fd,stroke:#1565c0,color:#000
    style Model fill:#e8f5e9,stroke:#2e7d32,color:#000
    style Training fill:#fff3e0,stroke:#e65100,color:#000
    style Inference fill:#f3e5f5,stroke:#6a1b9a,color:#000
```

### 1.2 Komponenty według warstw

| Warstwa | Klasa/Funkcja | Plik | Odpowiedzialność |
|---------|--------------|------|------------------|
| **Data** | `StockDataLoader` | [`data/stock_data.py`](../data/stock_data.py) | Pobieranie danych z Yahoo Finance |
| **Data** | `StockDataset` | [`data/stock_data.py`](../data/stock_data.py) | Tworzenie par context-target |
| **Data** | `create_dataloaders` | [`data/stock_data.py`](../data/stock_data.py) | Podział train/val/test |
| **Model** | `TimeSeriesEncoder` | [`models/encoder.py`](../models/encoder.py) | Enkoder czasowy (Conv1D + BiLSTM) |
| **Model** | `MomentumEncoder` | [`models/encoder.py`](../models/encoder.py) | Kopiia enkodera z aktualizacją EMA |
| **Model** | `RecurrentPredictor` | [`models/predictor.py`](../models/predictor.py) | Predyktor GRU w latent space |
| **Model** | `StockDecoder` | [`models/decoder.py`](../models/decoder.py) | Dekoder latent → ceny |
| **Model** | `RJEPA` | [`models/r_jepa.py`](../models/r_jepa.py) | Główny model łączący komponenty |
| **Training** | `JEPALoss` | [`training/loss.py`](../training/loss.py) | Funkcja straty JEPA |
| **Training** | `RJEPATrainer` | [`training/trainer.py`](../training/trainer.py) | Pętla treningowa |
| **Utils** | `compute_prediction_metrics` | [`utils/metrics.py`](../utils/metrics.py) | Metryki ewaluacyjne |
| **Utils** | `plot_predictions` | [`utils/metrics.py`](../utils/metrics.py) | Wizualizacja predykcji |

---

## 2. Przepływ danych

### 2.1 Przepływ treningowy

```mermaid
sequenceDiagram
    participant YF as Yahoo Finance
    participant DL as StockDataLoader
    participant DS as StockDataset
    participant M as RJEPA Model
    participant L as JEPA Loss
    participant O as Optimizer

    YF->>DL: Pobierz dane AAPL, MSFT, GOOGL
    DL->>DL: Inżynieria cech (RSI, MA, Vol)
    DL->>DL: Normalizacja (StandardScaler)
    DL->>DS: Podział train/val/test
    
    loop for each batch
        DS->>M: context [B, T, d], target [B, H, d]
        
        Note over M: Online Encoder
        M->>M: z_c = online_encoder(context)
        
        Note over M: Momentum Encoder (no grad)
        M->>M: z_t = momentum_encoder(target)
        
        Note over M: Predictor
        M->>M: Ẑ = predictor(z_c, H, Z_c)
        
        M->>L: predicted_latents=Ẑ, target_latent=z_t
        L->>L: L_pred = MSE(Ẑ, z_t)
        L->>L: L_var = variance_reg(Ẑ, z_t)
        L->>L: L_cov = covariance_reg(Ẑ, z_t)
        L->>L: L = L_pred + λ_var·L_var + λ_cov·L_cov
        
        L->>M: loss.backward()
        O->>M: optimizer.step()
        M->>M: update_momentum_encoder()
    end
```

### 2.2 Przepływ inferencyjny

```mermaid
sequenceDiagram
    participant DL as StockDataLoader
    participant M as RJEPA Model (eval)
    participant DEC as Decoder
    participant ENS as Ensemble
    participant VIZ as Visualization

    DL->>DL: Pobierz ostatnie T obserwacji
    DL->>M: context [1, T, d]
    
    Note over M: Online Encoder
    M->>M: z_c = online_encoder(context)
    M->>M: Z_c = online_encoder(context, return_sequence=True)
    
    Note over M: Predictor
    M->>M: Ẑ = predictor(z_c, H, Z_c)
    
    M->>DEC: predicted_latents=Ẑ
    DEC->>DEC: X̂ = decode_sequence(Ẑ)
    
    alt with ensemble
        M->>ENS: predict_with_noise(z_c, H, noise=0.05, N=10)
        ENS->>ENS: 10 trajectories
        ENS->>VIZ: mean, std, CI 95%
    else without ensemble
        DEC->>VIZ: X̂ [1, H, d]
    end
    
    VIZ->>VIZ: plot_predictions()
    VIZ->>VIZ: save predictions.csv
```

---

## 3. Komponenty systemu

### 3.1 StockDataLoader

```mermaid
classDiagram
    class StockDataConfig {
        +tuple~str~ tickers
        +str start_date
        +str end_date
        +int sequence_length
        +int prediction_horizon
        +tuple~str~ features
        +bool normalize
        +float train_split
        +float val_split
    }
    
    class StockDataLoader {
        -StockDataConfig config
        -StandardScaler _scaler
        -dict~str, DataFrame~ data
        -dict~str, ndarray~ processed_data
        +preprocess() ndarray
        +get_feature_names() list~str~
        +inverse_transform(ndarray) ndarray
        -_download_data()
        -_add_technical_features(DataFrame) DataFrame
    }
    
    class StockDataset {
        -Tensor data
        -int sequence_length
        -int prediction_horizon
        -list~int~ valid_indices
        +__len__() int
        +__getitem__(int) tuple~Tensor, Tensor~
    }
    
    StockDataConfig --> StockDataLoader : configures
    StockDataLoader --> StockDataset : creates
    StockDataset --> DataLoader : wraps
```

### 3.2 Model R-JEPA

```mermaid
classDiagram
    class TimeSeriesEncoder {
        -Sequential conv_layers
        -LSTM lstm
        -Sequential projection
        +forward(Tensor, bool) Tensor
        -_init_weights()
    }
    
    class MomentumEncoder {
        -TimeSeriesEncoder encoder
        -TimeSeriesEncoder momentum_encoder
        -float tau
        +update_momentum()
        +forward(Tensor, bool) Tensor
    }
    
    class RecurrentPredictor {
        -Sequential context_aggregator
        -GRU gru
        -Sequential output_projection
        +forward(Tensor, int, Tensor) Tensor
        +predict_with_noise(Tensor, int, float, int) Tensor
    }
    
    class StockDecoder {
        -Sequential decoder
        +forward(Tensor) Tensor
        +decode_sequence(Tensor) Tensor
    }
    
    class RJEPA {
        -TimeSeriesEncoder online_encoder
        -MomentumEncoder momentum_encoder
        -RecurrentPredictor predictor
        -StockDecoder decoder
        +forward(Tensor, Tensor, int) dict
        +predict(Tensor, int, bool, int, float) dict
        +update_momentum_encoder()
    }
    
    RJEPA *-- TimeSeriesEncoder : contains
    RJEPA *-- MomentumEncoder : contains
    RJEPA *-- RecurrentPredictor : contains
    RJEPA *-- StockDecoder : contains
    MomentumEncoder o-- TimeSeriesEncoder : wraps
```

### 3.3 Training

```mermaid
classDiagram
    class JEPALoss {
        -float variance_weight
        -float covariance_weight
        -float predictor_epsilon
        +forward(Tensor, Tensor, Tensor) dict
    }
    
    class RJEPATrainer {
        -RJEPA model
        -DataLoader train_loader
        -DataLoader val_loader
        -AdamW optimizer
        -CosineAnnealingLR scheduler
        -GradScaler scaler
        -list~float~ train_losses
        -list~float~ val_losses
        +train() RJEPA
        -_train_epoch() dict
        -_validate() dict
        -_save_checkpoint(bool)
        -_load_checkpoint(str)
        -_save_metrics()
    }
    
    RJEPATrainer --> RJEPA : trains
    RJEPATrainer --> JEPALoss : uses
    RJEPATrainer --> DataLoader : iterates
```

---

## 4. Tryb treningowy vs inferencyjny

| Aspekt | Trening | Inferencja |
|--------|---------|------------|
| **Model** | `model.train()` | `model.eval()` |
| **Encoder** | Online + Momentum | Tylko Online |
| **Target** | Wymagany (`target` tensor) | Nie wymagany |
| **Momentum update** | Tak (po każdym batchu) | Nie |
| **AMP** | Tak (GradScaler) | Nie |
| **Gradient** | Wymagany | `torch.no_grad()` |
| **Decoder** | Opcjonalny | Wymagany (do wizualizacji) |
| **Ensemble** | Nie | Opcjonalnie |
| **Confidence intervals** | Nie | Tak (przy ensemble) |

---

## 5. Diagram sekwencji — pełny cykl życia

```mermaid
sequenceDiagram
    participant U as User
    participant C as Config
    participant T as run_training.py
    participant D as Data
    participant M as Model
    participant TR as Trainer
    participant CK as Checkpoint

    U->>C: Edytuj config.yaml
    U->>T: python run_training.py
    
    T->>D: StockDataLoader(config)
    D->>D: Download data (yfinance)
    D->>D: Engineer features
    D->>D: Normalize
    
    T->>D: create_dataloaders()
    D->>T: train_loader, val_loader, test_loader
    
    T->>M: RJEPA(input_dim, ...)
    T->>TR: RJEPATrainer(model, loaders, config)
    
    loop for each epoch
        TR->>TR: _train_epoch()
        TR->>CK: Save best checkpoint
        TR->>TR: _validate()
        
        alt early stopping
            TR->>T: Stop training
        end
    end
    
    T->>T: plot_training_history()
    T->>U: Training complete!
    
    U->>U: python run_prediction.py --checkpoint ckpt.pt
    U->>M: Load checkpoint
    U->>M: predict(context)
    M->>U: predictions, confidence intervals
    U->>U: plot_predictions()
```

---

## 6. Zależności między modułami

```
run_training.py
    ├── data/stock_data.py
    │       └── yfinance, pandas, sklearn
    ├── models/r_jepa.py
    │       ├── models/encoder.py
    │       │       └── torch.nn (Conv1d, LSTM, Linear, LayerNorm)
    │       ├── models/predictor.py
    │       │       └── torch.nn (GRU, Linear, LayerNorm)
    │       └── models/decoder.py
    │               └── torch.nn (Linear, LayerNorm)
    ├── training/trainer.py
    │       ├── training/loss.py
    │       │       └── torch.nn.functional (mse_loss)
    │       └── torch.optim (AdamW, CosineAnnealingLR)
    └── utils/metrics.py
            └── matplotlib, numpy

run_prediction.py
    ├── data/stock_data.py
    ├── models/r_jepa.py
    └── utils/metrics.py
```
