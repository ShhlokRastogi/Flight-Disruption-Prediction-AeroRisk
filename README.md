# # ✈️ AeroRisk — Flight Disruption Prediction & Risk Scoring System

## 🧠 Problem Statement
Flight disruptions are **rare, asymmetric, highly imbalanced and hierarchical** events:
A single multiclass model often fails to learn these heterogeneous patterns effectively.
This structure makes direct multiclass learning difficult and unstable.

This project predicts **flight disruption outcomes** using structured airline operational data.  
The final system predicts one of four mutually exclusive outcomes:

- **Diverted** flights are extremely rare but operationally critical  
- **Cancelled** flights are often a consequence of extreme delays or systemic issues  
- **Delayed** flights dominate the dataset and overlap strongly with On-Time cases  
- **On Time** flights are the default majority outcome  

The work evolved through **multiple modeling paradigms**, extensive experimentation, and systematic failure analysis, finally converging on a **4-stage binary classification pipeline with a meta-classifier** and a **4-stage binary classification pipeline with a softmax and argmax implementation** which proved to be the most robust, interpretable, and deployment-ready solution.

This structure makes direct multiclass learning difficult and unstable.

## Dataset Summary

## 📊 Dataset Description

This project uses a large-scale historical **U.S. airline operations dataset** containing detailed information about scheduled and actual flight performance. The data captures **temporal, operational, route-level, and carrier-level characteristics** that influence flight disruptions.

---

### 📁 Data Scope

- **Time span:** Multiple years of historical flight records  
- **Scale:** Original dataset contains **~123 million rows**  
- **Training subsets used:** 1M – 4M rows (for experimentation and scalability)  
- **Final evaluation:** Balanced and stratified samples for fair comparison  

Each row represents **one scheduled flight**.

---

### 🎯 Target Variable

**`DisruptionType`**  
A categorical label representing the final operational outcome of a flight:

- **On Time** – Arrived without delay  
- **Delayed** – Arrived late  
- **Cancelled** – Flight did not operate  
- **Diverted** – Flight landed at a different airport  

This target is **highly imbalanced**, with *On Time* and *Delayed* flights dominating the dataset.

---

### ⏱ Temporal Features

These features capture when the flight occurs, which strongly influences congestion and delay propagation:

- `Year`
- `Month`
- `DayofMonth`
- `DayOfWeek`
- `CRSDepMin` – Scheduled departure time (minutes from midnight)
- `CRSArrMin` – Scheduled arrival time (minutes from midnight)
- `ScheduledElapsedTime` – Planned flight duration (minutes)

---

### 🛫 Route & Carrier Features

These describe where the flight operates and who operates it:

- `Origin` – Origin airport (IATA code)
- `Dest` – Destination airport (IATA code)
- `UniqueCarrier` – Airline carrier code
- `Distance` – Flight distance (miles)

Because of their **high cardinality**, these categorical variables are **not used directly** in modeling.

---

### 📈 Reliability Encodings (Derived Features)

High-cardinality categorical variables are converted into **Bayesian-smoothed reliability scores**, computed from historical data and stored externally as CSV files:

- `CarrierReliability`
- `OriginReliability`
- `DestReliability`

These represent historical **On-Time performance rates**, adjusted to avoid bias from rare observations.

---

### 🌅 Time-of-Day Encoding

Raw clock times are mapped into operational regimes to reduce noise:

| Encoding | Time Period |
|--------|------------|
| 0 | Morning (05–11) |
| 1 | Afternoon (11–17) |
| 2 | Evening (17–22) |
| 3 | Night (22–05) |

Features:
- `DepTimeOfDay_enc`
- `ArrTimeOfDay_enc`

This encoding captures airport congestion patterns and operational shifts more effectively than raw hours.

---

### 🧹 Features Removed During Preprocessing

Several columns were deliberately removed to prevent **data leakage**, reduce noise, or because they are **unavailable at prediction time**:

#### 🚫 Post-Event / Leakage Features
These are only known after the flight has occurred:
- `ArrDelay`
- `DepDelay`
- `CarrierDelay`
- `WeatherDelay`
- `NASDelay`
- `SecurityDelay`
- `LateAircraftDelay`
- `ActualElapsedTime`
- `AirTime`
- `TaxiIn`
- `TaxiOut`

#### 🚫 High-Cardinality Identifiers
These add noise and do not generalize well:
- `FlightNum`
- `TailNum`

#### 🚫 Redundant or Derived Columns
Removed to avoid multicollinearity or duplication:
- Raw `CRSDepTime` and `CRSArrTime` (replaced by minute-based representations)
- Raw categorical `Origin`, `Dest`, `UniqueCarrier` (replaced by reliability encodings)

#### 🚫 Low-Information or Inference-Only Columns
- `CancellationCode`
- Intermediate diagnostic columns used during EDA

---

### 🧹 Data Cleaning & Filtering Summary

- Missing values handled using route-level or global medians where applicable  
- Time features normalized to minute-based formats  
- Feature set aligned strictly between training and inference  
- All preprocessing steps replicated exactly during deployment  

---

### ⚠️ Key Challenges in the Data

- Extreme class imbalance  
- Strong overlap between **On Time** and **Delayed**  
- Rare but operationally critical **Diverted** events  
- High-cardinality categorical variables  
- Temporal dependency and delay propagation  

## Models Tried 

### 1. Direct Multiclass Classification ❌
Models tried:
- Random Forest
- XGBoost
- LightGBM
- CatBoost

**Issues:**
- Accuracy plateaued at ~30–55% on balanced data
- Severe confusion between Delayed / On Time / Cancelled
- Diverted class overwhelmed or overfit

**Conclusion:** Flat multiclass modeling does not respect the hierarchical nature of disruptions.


## 🧩APPROACH 1: 4-Stage Binary Pipeline + Meta Classifier

The final architecture decomposes the problem into **simpler, interpretable binary decisions**, then recombines them using a learned meta-model.
This idea was derived by the severe confusion between **On Time**, **Delayed**, and **Cancelled**

### 🔹 Stage 1 – Binary Base Models (trained on RandomForest)

Four binary classifiers are trained, each specializing in a **specific operational decision boundary**:

1. **Diverted vs Others**  
   Diverted vs others is binary classification type was chosen because diverted flights had more differentiablity from other labels in multiclass classifiers.

2. **On Time vs Delayed**  
   On Time vs Delayed binary classification type was chosen because on time and delayed flights had less differentiable features between each other as seen in multiclass classifiers.

3. **Delayed vs Cancelled**  
    Delayed vs Cancelled binary classification type was chosen because on time and delayed flights had less differentiable features between each other as seen in multiclass classifiers.

4. **On Time vs Cancelled**  
    On Time vs Cancelled binary classification type was chosen because on time and delayed flights had less differentiable features between each other as seen in multiclass classifiers.

Each model outputs a **probability**, not a hard label.

**Base model choice:**  
- Random Forest 
- Chosen for robustness, non-linearity handling, and probability stability
---
### 🔹 Stage 2 – Meta Classifier (trained on extra trees)

The probability outputs of the four binary models are used as **meta-features**:
[p_diverted, p_ontime_vs_delayed, p_delayed_vs_cancelled, p_ontime_vs_cancelled]

These meta-features are fed into a **meta-classifier** (Extra Trees / Random Forest), which learns how to:

- Resolve conflicting binary signals
- Weight model outputs by reliability
- Produce a final disruption class
  
This removes the need for:
- Hard thresholds
- Rule-based logic
- Manual prioritization
- 
## 🔍 How the Meta Classifier Improves Predictions
- Learns which binary models to trust in different scenarios  
- Smooths noisy probability outputs  
- Produces calibrated class probabilities  

The meta-classifier effectively acts as a **decision arbiter**, combining specialized expertise from each binary model.
## 📊 Output
For each flight, the model produces:

- Final predicted class  
  *(On Time / Delayed / Cancelled / Diverted)*  
- Probability score for each class  
- Confidence score (maximum probability)

## ✅ Advantages of This Architecture
- Reflects real-world disruption hierarchy    
- Stable on unseen data  
- Memory-safe inference (models loaded one at a time)  
- Easily extensible with new binary models
  
## METRICS
  TRAIN ACCURACY:72%
  TEST ACCURACY:74%
  ROC AUC SCORE(TEST):

