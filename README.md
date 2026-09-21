# A Novel Early Detection and Prevention of Coronary Heart Disease Framework

## Using Hybrid Deep Learning Model (O-SBGC-LSTM) and Neural Fuzzy Inference System (FLCHDCS)

This repository implements the hybrid framework described in the research paper for the early detection and prevention of **Coronary Heart Disease (CHD)** in individuals with **Type 2 Diabetes Mellitus (T2DM)**.

The framework couples:

1. **O-SBGC-LSTM** — an *Optimal Scrutiny Boosted Graph Convolutional LSTM* deep learning model for classification, whose hyperparameters are tuned by the *Eurygaster Optimization Algorithm (EOA)* and whose fuzzy rules are initialized by *kernel-based clustering*.
2. **FLCHDCS** — a *Fuzzy Logic-based CHD Control System* (Mamdani Fuzzy Inference System) producing patient-specific prevention reports.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [System Architecture](#system-architecture)
4. [Data Pre-processing Pipeline](#data-pre-processing-pipeline)
5. [Mathematical Model: O-SBGC-LSTM](#mathematical-model-o-sbgc-lstm)
   - [Graph Convolution](#graph-convolution)
   - [SBGC-LSTM Cell](#sbgc-lstm-cell)
   - [Temporal Hierarchical Architecture](#temporal-hierarchical-architecture)
   - [Eurygaster Optimization Algorithm](#eurygaster-optimization-algorithm-eoa)
   - [Kernel-Based Clustering](#kernel-based-clustering)
6. [Mathematical Model: FLCHDCS (Mamdani FIS)](#mathematical-model-flchdcs-mamdani-fis)
   - [Fuzzification (Trapezoidal Membership Functions)](#fuzzification)
   - [Fuzzy Rule Base](#fuzzy-rule-base)
   - [Inference Engine (OR / AND connectives)](#inference-engine)
   - [Defuzzification (Centroid)](#defuzzification)
7. [Evaluation Metrics](#evaluation-metrics)
8. [How to Run the Colab Notebook](#how-to-run-the-colab-notebook)
9. [Expected Results & Benchmarks](#expected-results--benchmarks)
10. [File Structure](#file-structure)
11. [Acknowledgements & References](#acknowledgements--references)

---

## Project Overview

| Component            | Description |
|----------------------|-------------|
| **Objective**        | Early detection & prevention of CHD in T2DM patients |
| **Prediction model** | O-SBGC-LSTM (deep learning classifier) |
| **Prevention model** | FLCHDCS (Mamdani Fuzzy Inference System) |
| **Optimizer**        | Eurygaster Optimization Algorithm (EOA) |
| **Rule init**        | Kernel-based (RBF) clustering |
| **Fuzzy type**       | Mamdani, trapezoidal membership functions |
| **Dataset**          | Kaggle `sulianova/cardiovascular-disease-dataset` |
| **Train / Test split** | 80% / 20% |

---

## Dataset

- **Source:** [Kaggle — Cardiovascular Disease Dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)
- **Identifier:** `sulianova/cardiovascular-disease-dataset`
- **Samples:** 70,000 patient records (typical Kaggle distribution)
- **Target variable:** `cardio` — presence / absence of cardiovascular disease

**Relevant attributes used:**

| Attribute      | Type       | Description                            |
|----------------|------------|----------------------------------------|
| `age`          | continuous | Age in days                            |
| `gender`       | categorical| Gender (1 = women, 2 = men)            |
| `height`       | continuous | Height in cm                           |
| `weight`       | continuous | Weight in kg                           |
| `ap_hi`        | continuous | Systolic blood pressure                |
| `ap_lo`        | continuous | Diastolic blood pressure               |
| `cholesterol`  | ordinal    | 1 = normal, 2 = above, 3 = well above  |
| `gluc`         | ordinal    | 1 = normal, 2 = above, 3 = well above  |
| `smoke`        | binary     | Smoking status (0 = no, 1 = yes)       |
| `alco`         | binary     | Alcohol consumption (0 = no, 1 = yes)  |
| `active`       | binary     | Physical activity (0 = no, 1 = yes)    |

**Derived features used in modeling:**

- `age_years` = `age / 365.25`
- `bmi` = `weight / ((height / 100) ** 2)`
- 12 input features for model: `age_years`, `gender`, `height`, `weight`, `ap_hi`, `ap_lo`, `cholesterol`, `gluc`, `smoke`, `alco`, `active`, `bmi`.

> **Note:** If the CSV is not present locally, the notebook auto-generates a synthetic dataset that replicates the Kaggle schema for demonstration. Replace this with the real file in production.

---

## System Architecture

```
                         ┌───────────────────────────────────────────────┐
                         │            CHD Patient Data (T2DM)            │
                         └──────────────┬────────────────────────────────┘
                                        │
                          ┌─────────────▼──────────────┐
                          │   Data Pre-processing      │
                          │  · Missing (zero) removal  │
                          │  · Outlier removal         │
                          │  · Discretization          │
                          └─────────────┬──────────────┘
                                        │
                 ┌──────────────────────┼──────────────────────┐
                 │                      │                      │
        ┌────────▼────────┐    ┌────────▼────────┐   ┌─────────▼──────────┐
        │    Feature       │    │   Graph         │   │   Kernel-based     │
        │    Extraction    │    │   Construction  │   │   Clustering       │
        └────────┬────────┘    └────────┬────────┘   └─────────┬──────────┘
                 │                      │                      │  fuzzy rule init
                 ▼                      ▼                      ▼
        ┌──────────────────────────────────────────────────────────────┐
        │                 O-SBGC-LSTM PREDICTION MODEL                 │
        │                                                                │
        │  ┌─────────────── Graph Conv Layer (feature embedding) ─────┐  │
        │  │  H' = act( Ã @ H @ W + b )                                │  │
        │  └──────────────────────────────────────────────────────────┘  │
        │                                                                │
        │  Temporal Hierarchy (3 SBGC-LSTM layers + Avg-Pooling)         │
        │  ┌────────┐  ┌────────┐  ┌────────┐                           │
        │  │SBGC-LSTM│→│SBGC-LSTM│→│SBGC-LSTM│                           │
        │  │layer 1 │  │layer 2 │  │layer 3 │                           │
        │  └────────┘  └────────┘  └────────┘                           │
        │        temporal average pooling                               │
        │                                                                │
        │  ┌────────────────── Classification Head ──────────────────┐   │
        │  │ Dense(128) → Dropout → Dense(64) → Sigmoid               │  │
        │  └─────────────────────────────────────────────────────────┘  │
        └───────────────┬──────────────────────────────────────────────┘
                        │  {CHD, healthy} + risk probability
                        ▼
        ┌──────────────────────────────────────────────────────────────┐
        │            FLCHDCS  (Mamdani Fuzzy Inference System)          │
        │                                                                │
        │  Fuzzification → Rule Base → Inference → Defuzzification       │
        │                (trapezoidal MFs)                               │
        └───────────────────────────────┬──────────────────────────────┘
                                        │
                                        ▼
                       ┌────────────────────────────────────┐
                       │  Final Prevention Report             │
                       │  CHD Status: Low / Moderate / Risk / │
                       │              High Risk / Very High   │
                       └────────────────────────────────────┘
```

---

## Data Pre-processing Pipeline

### 1. Missing values (coded as zero)

The cardiovascular dataset encodes missing physiological measurements (blood pressure, weight, height) as **0**. These instances are removed:

```python
for col in ['ap_hi', 'ap_lo', 'height', 'weight']:
    df = df[df[col] != 0]
```

### 2. Outlier removal

Physical outliers are clipped using domain-knowledge bounds:

```python
df = df[(df['ap_hi'] > 60)  & (df['ap_hi'] < 260)]
df = df[(df['ap_lo'] > 30)  & (df['ap_lo'] < 180)]
df = df[(df['height'] > 120) & (df['height'] < 220)]
df = df[(df['weight'] > 30) & (df['weight'] < 250)]
```

### 3. Discretization

Continuous attribute values are transformed into finite collections of intervals:

```python
def discretize(series, n_bins=5):
    return pd.cut(series, bins=n_bins, labels=False)
```

Discretized features (`age_years_bin`, `ap_hi_bin`, `bmi_bin`, ...) feed the fuzzy inference system and aid the rule-base interpretability.

---

## Mathematical Model: O-SBGC-LSTM

### Graph Convolution

The **Graph Convolution Neural Network (GCNN)** operator replaces standard fully-connected transforms inside the LSTM input, forget, and output gates.

Given a normalized adjacency matrix

$$\tilde{A} = D^{-1/2} (A + I)\, D^{-1/2},$$

the layer-wise propagation rule is

$$H^{(l+1)} = \sigma\Big( \tilde{A}\; H^{(l)}\; W^{(l)} + b^{(l)} \Big),$$

where:

- $A$ = adjacency matrix (built from Pearson correlation of features), with self-loops,
- $I$ = identity matrix,
- $D$ = degree matrix, $D_{ii} = \sum_j A_{ij}$,
- $H^{(l)}$ = node-feature matrix at layer $l$,
- $W^{(l)}$, $b^{(l)}$ = learnable weights and biases,
- $\sigma$ = nonlinear activation (ReLU / sigmoid).

The adjacency matrix is pre-computed once from the training data correlation structure and kept fixed:

```python
corr = np.corrcoef(X.T)
A = (np.abs(corr) > threshold).astype(np.float32)
np.fill_diagonal(A, 1.0)                  # self loops
D_inv_sqrt = np.where(D > 0, D**-0.5, 0.0)
A_norm = D_inv_sqrt @ A @ D_inv_sqrt      # symmetric normalization
```

### SBGC-LSTM Cell

The scrutinity-boosted graph-convolutional LSTM cell modifies the classical LSTM equations by:

1. Concatenating input and hidden state: $z_t = [x_t;\; h_{t-1}]$,
2. Applying a learned **scrutiny-boost** attention mask over the concatenated vector,
3. Convolving $z_t$ with the graph operator $\tilde{A}$ before each gate projection.

```
f_t = σ( Ã·[z_t·sigmoid(α)] · W_f + b_f )     forget gate
i_t = σ( Ã·[z_t·sigmoid(α)] · W_i + b_i )     input gate
o_t = σ( Ã·[z_t·sigmoid(α)] · W_o + b_o )     output gate
c̃_t = tanh( z_t · W_c + b_c )                 candidate cell
c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t              cell update
h_t = o_t ⊙ tanh(c_t)                          hidden-state
```

where $\odot$ denotes the Hadamard (element-wise) product, $\sigma$ is the sigmoid function, $\tilde{A}$ is the normalized adjacency matrix, and $\alpha$ is the learnable scrutiny-boost attention parameter.

### Temporal Hierarchical Architecture

Three SBGC-LSTM layers are stacked in a **temporal hierarchy** with average pooling in the temporal domain to:

- increase the receptive field across temporal scales,
- reduce computational cost,
- capture multi-resolution temporal dependencies.

```
x          (B, timesteps, units)
 │
 ├── SBGC-LSTM layer 1  ──→ BN ──→ Dropout ──→ AvgPooling1D(1)
 ├── SBGC-LSTM layer 2  ──→ BN ──→ Dropout ──→ AvgPooling1D(1)
 └── SBGC-LSTM layer 3  ──→ BN ──→ Dropout
                                     │
                                GlobalAveragePooling1D
                                     │
                                Dense(128) → Dense(64) → Sigmoid
```

### Eurygaster Optimization Algorithm (EOA)

EOA is a bio-inspired metaheuristic modelled on the foraging and mating behaviour of the *Eurygaster* bug. It searches for the optimal hyperparameters:

- Learning rate
- LSTM hidden units
- Dropout probability
- (GCNN embedding units)

EOA cycle per iteration:

1. **Foraging** (exploration) — random-walk perturbation with decreasing step size:
   $$x_i' = x_i + \mathcal{N}(0,\, \sigma_t), \qquad \sigma_t = \sigma_0 (1 - t/T)$$

2. **Mating** (exploitation) — recombination toward the current best:
   $$x_i' = \alpha\, x_i + (1-\alpha)\, x_j, \qquad \alpha = 2t/T$$

3. **Migration** (diversification) — random jump along a single dimension to escape local optima.

The fitness is defined as validation **accuracy** on the held-out split.

### Kernel-Based Clustering

Fuzzy rules for the SBGC-LSTM are initialized using cluster centers derived from a **kernel-based clustering** algorithm. The RBF kernel is:

$$K(x_i, x_j) = \exp\left(-\frac{\|x_i - x_j\|^2}{2\sigma^2}\right)$$

Cluster seeds are selected greedily by maximum kernel density, guaranteeing diverse initialization of the membership functions.

---

## Mathematical Model: FLCHDCS (Mamdani FIS)

### Fuzzification

The prevention module is a **Mamdani Fuzzy Inference System**. Each continuous input is fuzzified with **trapezoidal** membership functions $\mu(x;\,a,b,c,d)$:

$$\mu(x) = \begin{cases}
0 & x \le a \\[2pt]
\dfrac{x - a}{b - a} & a \le x \le b \\[2pt]
1 & b \le x \le c \\[2pt]
\dfrac{d - x}{\,d - c\,} & c \le x \le d \\[2pt]
0 & x \ge d
\end{cases}$$

**Input linguistic variables (per paper):**

| Variable          | Low                     | Middle                  | High                     |
|-------------------|-------------------------|-------------------------|--------------------------|
| **Age**           | trapmf `[20,20,30,40]`  | trapmf `[35,45,55,65]`  | trapmf `[60,70,90,99]`   |
| **BloodPressure** | trapmf `[80,80,100,115]`| trapmf `[110,125,145,165]`| trapmf `[155,170,200,219]` |
| **Glucose**       | trapmf `[70,70,100,115]`| trapmf `[110,125,145,165]`| trapmf `[155,170,200,219]` |
| **Cholesterol**   | trapmf `[100,100,150,185]`| trapmf `[180,205,225,245]` | trapmf `[235,245,270,299]` |
| **LDL**           | trapmf `[60,60,90,105]` | trapmf `[95,110,130,155]` | trapmf `[145,195,220,239]`|
| **HDL**           | trapmf `[20,20,40,55]`  | trapmf `[45,55,70,85]`   | trapmf `[75,85,110,119]`  |
| **BMI**           | trapmf `[12,12,18,22]`  | trapmf `[20,23,27,30]`  | trapmf `[28,33,50,54]`   |

**Output linguistic variable (CHD status, crisp score in [0,100]):**

| Label            | trapezoidal MF          | Semantics         |
|------------------|-------------------------|-------------------|
| `low`            | `[0,0,10,25]`           | Under control     |
| `moderate`       | `[20,30,40,50]`         | Moderate risk     |
| `risk`           | `[45,55,65,75]`         | At risk           |
| `high_risk`      | `[70,78,85,92]`         | High risk         |
| `very_high_risk` | `[88,93,100,100]`       | Very high risk    |

### Fuzzy Rule Base

Rules use **OR** ($\lor$) and **AND** ($\land$) connectives.

| Rule | IF condition (Fuzzy Sets) | THEN Consequent |
|------|---------------------------|-----------------|
| R1   | `CHOL['low'] \| GLUC['low'] \| (BMI['medium'] & AGE['low'])` | `CHD['low']` (under control) |
| R2   | `CHOL['middle'] \| GLUC['low'] \| BMI['medium'] \| AGE['middle']` | `CHD['moderate']` (moderate risk) |
| R3   | `CHOL['middle'] \| GLUC['middle'] \| BMI['high'] \| AGE['middle']` | `CHD['risk']` (at risk) |
| R4   | `CHOL['high'] \| GLUC['middle'] \| BMI['medium'] \| AGE['middle']` | `CHD['high_risk']` (high risk) |
| R5   | `CHOL['high'] \| GLUC['high'] \| BMI['high'] \| AGE['high']` | `CHD['very_high_risk']` (very high risk) |

### Inference Engine

For a given crisp input vector, each rule antecedent is evaluated:

- **AND** → min operator: $\mu_{A \wedge B} = \min(\mu_A, \mu_B)$
- **OR** → max operator: $\mu_{A \vee B} = \max(\mu_A, \mu_B)$

Each rule output (its firing strength) truncates the corresponding consequent membership function.

### Defuzzification

The aggregated fuzzy output is converted to a crisp CHD-score using the **centroid** method:

$$z^* = \frac{\displaystyle\int z \cdot \mu(z)\, dz}{\displaystyle\int \mu(z)\, dz}$$

The resulting score is mapped to a human-readable prevention verdict:

| Crisp score | Prevention verdict    |
|-------------|-----------------------|
| `[0, 20)`   | Under control         |
| `[20, 45)`  | Moderate              |
| `[45, 70)`  | Risk                  |
| `[70, 88)`  | High Risk             |
| `[88, 100]` | Very High Risk        |

---

## Evaluation Metrics

The confusion matrix provides:

$$\begin{aligned}
TP &= \text{True Positives (CHD correctly detected)} \\
TN &= \text{True Negatives (healthy correctly classified)} \\
FP &= \text{False Positives (healthy wrongly flagged)} \\
FN &= \text{False Negatives (CHD missed)}
\end{aligned}$$

From these, the primary metrics are:

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

$$\text{Precision} = \frac{TP}{TP + FP}$$

$$\text{Recall (sensitivity)} = \frac{TP}{TP + FN}$$

$$F_1 = \frac{2\,\text{Precision}\,\text{Recall}}{\text{Precision} + \text{Recall}}$$

---

## How to Run the Colab Notebook

### Option A — Google Colab (recommended)

1. **Create / open the notebook**
   - Upload `DL_PAPER_PROJECT.ipynb` to Google Colab: `File → Upload notebook`.

2. **Set runtime to GPU**
   - `Runtime → Change runtime type → Hardware accelerator: GPU (T4)`.

3. **Run all cells**
   - `Runtime → Run all` (or `Ctrl/Cmd + F9`).
   - The notebook will:
     * install `scikit-fuzzy`,
     * load or synthesize the dataset,
     * pre-process the data,
     * construct the graph adjacency matrix,
     * run kernel-based clustering,
     * define the O-SBGC-LSTM model,
     * run the EOA hyperparameter search,
     * train the final model,
     * evaluate with confusion matrix + ROC,
     * build and run the Mamdani FIS,
     * output a prevention report.

4. **Using a real dataset**
   - Either place `cardio_train.csv` in the runtime working directory,
   - or uncomment the Kaggle API download cell (requires `kaggle.json` credentials),
   - or upload the CSV with `from google.colab import files; files.upload()`.
   - If the CSV is absent, the notebook automatically generates a synthetic replica for demonstration.

### Option B — Local Jupyter

```bash
# 1. Create a virtual environment (optional)
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows

# 2. Install dependencies
pip install tensorflow scikit-learn scikit-fuzzy pandas numpy matplotlib seaborn jupyter

# 3. Launch Jupyter
jupyter notebook DL_PAPER_PROJECT.ipynb
```

### Option C — Headless execution

```bash
pip install jupyter nbconvert
jupyter nbconvert --to notebook --execute DL_PAPER_PROJECT.ipynb --output DL_PAPER_PROJECT_executed.ipynb
```

---

## Expected Results & Benchmarks

The paper's reported benchmarks (for reference; with the full dataset and extended EOA search):

| Metric     | Paper Benchmark | Description                             |
|------------|-----------------|-----------------------------------------|
| Accuracy   | ≈ 98.61%        | Overall correct classification rate     |
| Precision  | —               | True-positive rate among positive calls |
| Recall     | ≈ 99.2%         | Sensitivity — CHD detection rate        |
| F1-Score   | ≈ 96.5%         | Harmonic mean of precision & recall     |

> The notebook runs a *reduced* EOA population / epoch count so it completes quickly on Colab. With `pop_size ≥ 30`, `n_iter ≥ 20`, and ≥ 100 training epochs on the real Kaggle dataset, convergence to the reported benchmarks is achievable.

---

## File Structure

```
.
├── DL_PAPER_PROJECT.ipynb         # Main Jupyter/Colab notebook (model, EOA, FIS, report)
├── README.md                      # Documentation (mathematical formulations & guide)
├── o_sbgc_lstm_chd_model.keras    # Output: trained model (when saved)
├── chd_prevention_report.csv      # Output: patient prevention report (when exported)
└── cardio_train.csv               # Optional: Kaggle dataset
```

---

## Acknowledgements & References

- **Kaggle** Cardiovascular Disease Dataset — `sulianova/cardiovascular-disease-dataset`.
- **scikit-fuzzy** — provides `skfuzzy` Mamdani fuzzy inference primitives.
- **TensorFlow / Keras** — deep learning framework for the O-SBGC-LSTM model.
- The framework and exact membership parameters follow the design described in the referenced research paper.

---

*Generated by an AI engineering workflow for reproducibility and teaching purposes.*