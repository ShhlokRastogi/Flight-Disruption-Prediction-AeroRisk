# ✈️ AeroRisk — Flight Disruption Prediction & Risk Scoring System

---

## 🧠 Problem Statement

Flight disruptions are **rare, asymmetric, highly imbalanced, and hierarchical** events.  
A single multiclass classifier often fails to learn these heterogeneous patterns effectively, leading to unstable predictions and poor generalization.

This project predicts **flight disruption outcomes** using structured airline operational data.  
The final system predicts one of four mutually exclusive outcomes:

- **Diverted** — extremely rare but operationally critical  
- **Cancelled** — often a consequence of extreme delays or systemic failures  
- **Delayed** — common and strongly overlapping with On-Time cases  
- **On Time** — default majority outcome  

Through extensive experimentation, failure analysis, and iterative redesign, the project converged on two robust solutions:

- **A 4-stage binary classification pipeline with hard labeling + meta-classifier**
- **A 4-stage binary pipeline with softmax + argmax aggregation**

These approaches proved to be the most **robust, interpretable, and deployment-ready**.

---

## 📊 Dataset Description

This project uses a large-scale historical **U.S. airline operations dataset** containing detailed information about scheduled and actual flight performance. The data captures **temporal, operational, route-level, and carrier-level characteristics** that influence flight disruptions.

---

### 📁 Data Scope

- **Time span:** Multiple years of historical flight records  
- **Scale:** Original dataset contains **~123 million rows**  
- **Training subsets used:** 1M rows 
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

- `Year`
- `Month`
- `DayofMonth`
- `DayOfWeek`
- `CRSDepMin` – Scheduled departure time (minutes from midnight)
- `CRSArrMin` – Scheduled arrival time (minutes from midnight)
- `ScheduledElapsedTime` – Planned flight duration (minutes)

---

### 🛫 Route & Carrier Features

- `Origin` – Origin airport (IATA code)
- `Dest` – Destination airport (IATA code)
- `UniqueCarrier` – Airline carrier code
- `Distance` – Flight distance (miles)

Due to **high cardinality**, these categorical variables are **not used directly** in modeling.

---

### 📈 Reliability Encodings (Derived Features)

To handle high-cardinality categorical variables, **Bayesian-smoothed reliability encodings** were computed and stored as CSV files:

- `CarrierReliability`
- `OriginReliability`
- `DestReliability`

These represent historical **On-Time performance rates**, smoothed to avoid bias from rare observations.

---

### 🌅 Time-of-Day Encoding

Raw clock times were converted into operational regimes:

| Encoding | Time Period |
|--------|------------|
| 0 | Morning (05–11) |
| 1 | Afternoon (11–17) |
| 2 | Evening (17–22) |
| 3 | Night (22–05) |

Features:
- `DepTimeOfDay_enc`
- `ArrTimeOfDay_enc`

This encoding captures congestion and operational shifts better than raw hours.

---

### 🧹 Features Removed During Preprocessing

To prevent **data leakage**, reduce noise, and ensure inference-time availability, the following features were removed:

#### 🚫 Post-Event / Leakage Features
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
- `FlightNum`
- `TailNum`

#### 🚫 Redundant or Replaced Features
- Raw `CRSDepTime`, `CRSArrTime` (replaced by minute-based formats)
- Raw `Origin`, `Dest`, `UniqueCarrier` (replaced by reliability encodings)

#### 🚫 Low-Information Columns
- `CancellationCode`
- Intermediate EDA-only features

---

### ⚠️ Key Challenges in the Data

- Extreme class imbalance  
- Strong overlap between **On Time** and **Delayed**  
- Rare but operationally critical **Diverted** events  
- High-cardinality categorical variables  
- Temporal dependency and delay propagation  

---

## 🔬 Models Tried

### 1. Direct Multiclass Classification ❌

Models evaluated:
- Random Forest
- XGBoost
- LightGBM
- CatBoost

**Issues Observed:**
- Accuracy plateaued at ~30–55% on balanced data  
- Severe confusion between *On Time*, *Delayed*, and *Cancelled*  
- Diverted class either ignored or overfit  

**Conclusion:** Flat multiclass modeling does not respect the hierarchical nature of flight disruptions.

---

## 🧩 APPROACH 1: 4-Stage Binary Pipeline + Hard Labeling + Meta Classifier

This architecture decomposes the problem into **simpler, interpretable binary decisions**, then recombines them intelligently.

### 🔹 Stage 1 – Binary Base Models (Random Forest)

Four binary classifiers were trained, each targeting a **specific operational boundary**:

1. **Diverted vs Others**  
   Chosen because diverted flights showed strong separability from other outcomes.

2. **On Time vs Delayed**  
   Chosen due to heavy feature overlap observed in multiclass classifiers.

3. **Delayed vs Cancelled**  
   Helps distinguish recoverable delays from true cancellations.

4. **On Time vs Cancelled**  
   Captures edge cases where cancellations occur without long delays.

Each model outputs **both probabilities and hard labels**.

---

### 🔹 Hard Labeling Logic (Intermediate Decision Layer)

Before aggregation, **hard predictions** are evaluated:

- If **Diverted vs Others** predicts *Diverted*, the flight is immediately labeled **Diverted**
- Otherwise, hard labels from the remaining three models are compared
- The class with **majority support** becomes the provisional outcome

This step:
- Improves recall for rare events  
- Reduces ambiguity in high-confidence cases  
- Mimics real-world operational decision logic  

**Metrics:**
- Train Accuracy: **71–72%**
- Test Accuracy: **~70%**

---

### 🔹 Stage 2 – Meta Classifier (Extra Trees)

The **probability outputs** of the four binary models are used as meta-features:
A **meta-classifier (Extra Trees)** learns how to:
- Resolve conflicts between binary predictions
- Weight model outputs by reliability
- Produce a stable final class prediction

This removes the need for:
- Hard thresholds
- Manual rule tuning
- Brittle if–else logic

---

## 📊 Final Output

For each flight, the system produces:
- Final predicted disruption type  
- Probability score for each class  
- Confidence score (maximum probability)  

---

## ✅ Final Performance

- **Train Accuracy:** ~72%  
- **Test Accuracy:** ~74%  
- **ROC-AUC (Test):** Improved significantly over direct multiclass models  
- Stable full-coverage predictions on unseen data  

---

## 🚀 Key Advantages

- Reflects real-world disruption hierarchy  
- Strong rare-class detection  
- Interpretable decision flow  
- Memory-safe inference (models loaded one-by-one)  
- Easily extensible with new binary models  

---

## 🏁 Final Verdict

The **4-stage binary pipeline with hard labeling and a meta-classifier** emerged as the most **accurate, stable, and deployable** solution, outperforming all direct multiclass and naive ensemble approaches tried during the project.



