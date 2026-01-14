# ✈️ **AeroRisk**

### *Flight Disruption Prediction & Risk Scoring System*

---

## 🧠 Problem Statement

Flight disruptions are **rare, asymmetric, highly imbalanced, and hierarchical** events.
A single flat multiclass classifier struggles to learn these heterogeneous patterns, often resulting in unstable predictions and poor generalization.

**AeroRisk** predicts **flight disruption outcomes** from structured airline operational data.
Each flight is classified into **one of four mutually exclusive outcomes**:

| Outcome         | Description                                      |
| --------------- | ------------------------------------------------ |
| ✈️ **Diverted** | Extremely rare but operationally critical        |
| ❌ **Cancelled** | Often follows severe delays or systemic failures |
| ⏱ **Delayed**   | Common, with strong overlap with On-Time         |
| ✅ **On Time**   | Majority / default outcome                       |

After extensive experimentation and failure analysis, the project converged on **two robust architectures**:

* 🔹 **4-stage binary pipeline with hard labeling + meta-classifier**
* 🔹 **4-stage binary OvR ensemble with softmax + argmax aggregation**

These approaches are **interpretable, robust, and deployment-ready**.

---

## 📊 Dataset Overview

Large-scale historical **U.S. airline operations data**, capturing temporal, operational, route-level, and carrier-level signals influencing disruptions.

### 📁 Data Scope

* **Time span:** Multiple years
* **Original scale:** ~123 million flights
* **Training subset:** 1M rows
* **Final evaluation:** Balanced & stratified samples each label has 0.25M rows

Each row corresponds to **one scheduled flight**.

---

## 🎯 Target Variable

**`DisruptionType`** (highly imbalanced):

* On Time
* Delayed
* Cancelled
* Diverted

---

## 🧬 Feature Engineering

### ⏱ Temporal Features

* Year, Month, DayofMonth, DayOfWeek
* CRSDepMin, CRSArrMin
* ScheduledElapsedTime

---

### 🛫 Route & Carrier (Encoded)

Raw high-cardinality identifiers are **not used directly**:

* Origin, Dest, UniqueCarrier

Instead, **Bayesian-smoothed reliability encodings** are applied:

* CarrierReliability
* OriginReliability
* DestReliability

These represent historical **On-Time performance rates**, smoothed to reduce bias.

---

### 🌅 Time-of-Day Encoding

| Code | Period            |
| ---- | ----------------- |
| 0    | Morning (05–11)   |
| 1    | Afternoon (11–17) |
| 2    | Evening (17–22)   |
| 3    | Night (22–05)     |

Features:

* DepTimeOfDay_enc
* ArrTimeOfDay_enc

---

### 🧹 Features Removed (Leakage Prevention)

**Post-event / leakage:**
ArrDelay, DepDelay, WeatherDelay, NASDelay, TaxiIn/Out, AirTime, ActualElapsedTime, etc.

**Identifiers:**
FlightNum, TailNum

**Redundant:**
Raw CRS times, raw categorical IDs

---

## ⚠️ Key Data Challenges

* Extreme class imbalance
* Strong On-Time vs Delayed overlap
* Rare but critical Diverted events
* High-cardinality categorical features
* Temporal dependency & delay propagation

---

## ❌ Direct Multiclass Modeling (Baseline)

Models tested:

* Random Forest
* XGBoost
* LightGBM
* CatBoost

**Observed failures:**

* Accuracy capped at **30–55%** (balanced data)
* Severe confusion between On Time / Delayed / Cancelled
* Diverted either ignored or overfit

➡️ **Conclusion:** Flat multiclass modeling fails to capture disruption hierarchy.

![Multiclass Failure](https://github.com/user-attachments/assets/7c8c95e7-4af0-4471-afbc-244af585e6ce)

---

## 🧩 APPROACH 1 — 4-Stage Binary Pipeline + Meta Classifier

### 🔹 Stage 1: Binary Base Models (Random Forest)

| Binary Task          | Accuracy | Purpose                        |
| -------------------- | -------- | ------------------------------ |
| Diverted vs Others   | 86%      | Early capture of rare events   |
| On Time vs Delayed   | 60%      | Resolve heavy overlap          |
| Delayed vs Cancelled | 74%      | Distinguish recoverable delays |
| On Time vs Cancelled | 74%      | Edge-case separation           |

Each model outputs **probabilities + hard labels**.

---

### 🔹 Hard Labeling Logic

Decision rules:

* If *Diverted* → immediately classify
* Else, majority vote from remaining binaries

**Benefits:**

* Boosts rare-class recall
* Reduces ambiguity
* Mirrors real-world operational logic

**Accuracy:** ~70%

![Hard Labeling](https://github.com/user-attachments/assets/f2c2d329-512c-4741-a006-59dddd6ba7f2)

---

### 🔹 Stage 2: Meta Classifier (Extra Trees)

Meta-features = **probabilities from all binary models**.

Learns to:

* Resolve conflicts
* Weight model reliability
* Produce stable final prediction

Removes brittle thresholds & manual rules.

---

## 📈 Final Output (Approach 1)

For each flight:

* Final disruption label
* Class-wise probabilities
* Confidence score

**Performance:**

* Train Accuracy: ~72%
* Test Accuracy: ~74%

![Meta Results](https://github.com/user-attachments/assets/74a38b62-0bd4-4c45-b942-53d9cae9aad2)
![Confusion](https://github.com/user-attachments/assets/214693de-3f0f-4852-9395-eda808904cc5)

---

## 🚀 APPROACH 2 — OvR Ensemble with Probabilistic Coupling

### 🧠 Why OvR?

* Independent learning per disruption type
* Class-specific imbalance handling
* Interpretable decision boundaries

---

### 🌳 Base OvR Models (Tree Ensembles)

Each model predicts **P(class vs rest)** using Extra Trees Classifier

---

### 📊 OvR Model Performance

**Diverted vs Rest:** ROC-AUC ≈ 1.00
**Cancelled vs Rest:** ROC-AUC ≈ 0.92
**Delayed / On-Time vs Rest:** ROC-AUC ≈ 0.77

---

### 🔄 Probability Coupling (Softmax)

```
P(classᵢ) = exp(scoreᵢ) / Σ exp(scoreⱼ)
```

Ensures:

* Comparable probabilities
* Sum-to-one constraint
* Valid probabilistic metrics

---

### 🏁 Final Prediction

```
argmax(P(Cancelled), P(Diverted), P(Delayed), P(On Time))
```

**Accuracy:** 70%

---

## ⚠️ Risk Scoring

```
Risk Score = 1 − P(On Time)
```

Enables:

* Risk ranking
* Threshold-free prioritization
* Operational planning

---

## 📊 Final Evaluation (OvR)

| Class     | Precision | Recall | F1   |
| --------- | --------- | ------ | ---- |
| Cancelled | 0.90      | 0.65   | 0.76 |
| Diverted  | 0.99      | 1.00   | 1.00 |
| Delayed   | 0.49      | 0.58   | 0.54 |
| On Time   | 0.50      | 0.55   | 0.52 |

<img width="601" height="470" alt="image" src="https://github.com/user-attachments/assets/26a54fe4-5b8b-4394-9497-ad42d5e8e18c" />


* **Accuracy:** 69%
* **Macro / Weighted ROC-AUC:** 0.88
  <img width="790" height="590" alt="image" src="https://github.com/user-attachments/assets/5959dffc-d11e-456b-b21b-682fd0e70b00" />
  <img width="790" height="590" alt="image" src="https://github.com/user-attachments/assets/83d8886e-948d-462d-8022-880527769f16" />



---

## 🧩 Memory-Safe Inference

Designed for **low-memory environments**:

* Load one model at a time
* Predict & release immediately
* Streamlit-safe deployment

---

## ✅ Final Takeaways

✔ Reflects real-world disruption hierarchy
✔ Strong rare-event detection
✔ Transparent decision flow
✔ Robust & extensible system

**AeroRisk** prioritizes **correctness, interpretability, and deployability** — making it suitable for real-world airline operations.
