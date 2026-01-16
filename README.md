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
The dataset is taken from kaggle follow this link to find data -
https://www.kaggle.com/datasets/bulter22/airline-data

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

**OvR Accuracy Summary**

* **Diverted vs Rest**

  * Train Accuracy: 99%
  * Test Accuracy: 99%
  * ROC-AUC Test  : 0.99
  * Precision     : 0.99
  * Recall        : 0.99
  * F1-Score      : 0.99

* **Cancelled vs Rest**

  * Train Accuracy: 99%
  * Test Accuracy: 90%
  * ROC-AUC Test  : 0.91
  * Precision     : 0.93
  * Recall        : 0.63
  * F1-Score      : 0.75

* **Delayed vs Rest**

  * Train Accuracy: 93%
  * Test Accuracy: 71%
  * ROC-AUC Test  : 0.76
  * Precision     : 0.44
  * Recall        : 0.63
  * F1-Score      : 0.52

* **On-Time vs Rest**

  * Train Accuracy: 93%
  * Test Accuracy: 71%
  * ROC-AUC Test  : 0.76
  * Precision     : 0.44
  * Recall        : 0.60
  * F1 score : 0.51
<img width="627" height="470" alt="image" src="https://github.com/user-attachments/assets/4cad6627-c155-4a08-a6c4-a5deeaa9a9ad" />
<img width="627" height="470" alt="image" src="https://github.com/user-attachments/assets/54631ec9-1881-498e-9f00-4f4a5fcf55b5" />
<img width="629" height="470" alt="image" src="https://github.com/user-attachments/assets/39e727c2-3740-4275-8d1c-f9b6acb07e22" />
<img width="637" height="470" alt="image" src="https://github.com/user-attachments/assets/af5fd40c-d509-496a-b12c-36b730025039" />

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




## 🌟 Distinguishing Features

AeroRisk is not a typical *train-a-model-and-predict* project. It reflects **real-world ML system design**, shaped by constraints, failures, and operational needs.

---
## 🧩 Memory-Safe Inference

Designed for **low-memory environments**:

* Load one model at a time
* Predict & release immediately
* Streamlit-safe deployment

---

### 🧠 System-First Design

* Built as a **complete ML system**, not a standalone model
* Clear separation of feature engineering, modeling, inference, calibration, and risk scoring
* Mirrors how production ML systems are actually deployed

---

### 🧩 Structured Problem Decomposition

* Replaces flat multiclass prediction with **hierarchical binary decisions**
* Uses One-vs-Rest and staged pipelines to handle heterogeneous disruption types
* Reduces class confusion and improves rare-event handling

---

### 🎯 Specialized, Interpretable Models

* Each binary model solves **one clear operational question**
* Avoids forcing a single model to learn contradictory patterns
* Improves interpretability and debugging

---

### 🔄 Probability-Aware Decision Making

* Explicit **probability coupling (softmax)** for OvR models
* Probabilistically correct multiclass inference
* Enables valid ROC-AUC and confidence-based decisions

---

### ⚠️ Risk Scoring Over Hard Labels

* Outputs a **continuous disruption risk score**, not just a class
* Supports ranking, prioritization, and threshold-free decisions
* Better aligned with real operational workflows

---

### 🧹 Deployment-Conscious Feature Engineering

* No leakage-prone or post-event features
* High-cardinality categories handled via **Bayesian reliability encodings**
* All features available **before flight departure**

---

### 💾 Memory-Safe by Design

* Models loaded sequentially and released immediately
* Designed for low-memory environments (e.g., Streamlit Cloud)
* A production constraint often ignored in academic projects

---

### 🔬 Failure-Driven Iteration

* Architecture evolved from **documented failures**
* Each design choice is empirically motivated
* Demonstrates engineering maturity, not just performance

---

## ⚠️ Limitations & Trade-offs

* **Accuracy Ceiling (~70–75%)**
  Strong overlap between *On Time*, *Delayed*, and *Cancelled* limits separability. The model predicts **risk**, not certainty.

* **Pre-Departure Signals Only**
  No real-time weather, ATC, or aircraft rotation data, limiting last-minute disruption prediction.

* **Independent OvR Modeling**
  One-vs-Rest classifiers are trained independently and do not model transitions between disruption types.

* **No Temporal Dependency Modeling**
  Flights are treated independently; delay propagation and aircraft rotation chains are not captured.

* **Reliance on Historical Reliability Encodings**
  Carrier and airport reliability assume historical stability; sudden operational shifts may lag.

* **Probability Calibration Assumptions**
  Softmax coupling assumes comparable confidence scales across OvR models, introducing minor calibration risk.

* **Memory-Heavy Ensembles**
  Tree-based models require careful memory management, limiting low-resource cloud deployment.

* **Manual Retraining Required**
  No automated drift detection; periodic retraining is required as airline operations evolve.

---
## 👥 Potential Deployement Sectors

- **Airline Operations Control Centers (OCC)**
  - Predict and manage delays, cancellations, and diversions before departure

- **Airport Operations Teams**
  - Plan gates, ground staff, and resources during peak congestion periods

- **Airline Customer Experience Systems**
  - Proactively notify passengers and trigger rebooking workflows

- **Network Planning & Revenue Teams**
  - Identify unreliable routes, carriers, and time slots for long-term planning

- **Cargo & Logistics Operators**
  - Assess disruption risk for time-sensitive shipments

- **Risk, Compliance & Analytics Teams**
  - Audit disruption patterns and operational reliability

- **Real-Time Decision Support Systems**
  - Integrate as a risk-scoring API for operational dashboards

## **🧠👨‍🍳HOW TO USE THIS REPO**
##  ⬇️Download Model , Encodings and Data from this google drive link because of uploadation size issues
- https://drive.google.com/drive/folders/1OHa2DibO5WJtq-G_UaqDgOo1DN-0eILO?usp=sharing

## verify paths before using the models




