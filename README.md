# RetailPulse AI 📈
### Multi-Store Weekly Sales Analytics & Scenario Forecasting System

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-F7931E?logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Portfolio--Ready-0d9488)
![Academic Level](https://img.shields.io/badge/Academic-2nd%20Year%20Data%20Science-blue)

---

## 🌟 Overview
**RetailPulse AI** is an end-to-end Machine Learning and Time-Series Analytics application designed to forecast weekly sales across **45 retail outlets** (historical Walmart dataset). Built from the perspective of an ambitious **2nd-year undergraduate Data Science student**, this system demonstrates rigorous time-series principles, lookahead bias prevention, dynamic scenario simulation, and an intuitive, executive-ready dashboard.

---
Streamlit web link
https://retailpulse-ai1.streamlit.app/

Kaggle Dataset Link
https://www.kaggle.com/datasets/yasserh/walmart-dataset?resource=download

## 🎯 What It Does
* 📊 **Executive Overview:** Real-time business KPIs ($ total revenue, weekly averages, volume leaders) and annual seasonal surge analysis.
* 🏪 **Store Deep Dive:** Interactive per-store analytics featuring 4-week smoothed moving averages, holiday impact flags, and monthly seasonality profiles.
* 🔮 **Sales Forecast Simulator:** Interactive 1–12 week recursive forecasting with **dynamic US retail holiday auto-detection** and macroeconomic scenario planning (Temperature, Fuel, CPI, Unemployment).
* 🧪 **Model Performance Lab:** Transparent leaderboard benchmarking **Naive (Lag 1)**, **Seasonal Naive (Lag 52)**, **Random Forest**, and **HistGradientBoosting** across Holdout MAE, RMSE, MAPE, and R².
* 🎓 **Project Guide & Viva Prep:** Comprehensive walkthrough of data science methodology, time-series safeguards, and limitations for academic presentation.

---

## 📐 Time-Series Safeguards & Methodology

1. **Strict Chronological Holdout:**
   * The final **20 consecutive weeks** across all 45 stores are held out as the test set. Random K-fold splitting is strictly avoided to prevent future data from leaking into the past.
2. **Shifted Rolling Windows (`shift(1)`):**
   * Rolling averages (`Rolling_Mean_4`) and standard deviations (`Rolling_Std_4`) are calculated strictly on values up to $t-1$, ensuring zero lookahead bias.
3. **Store Scale Encoding (`Store_Avg_Sales`):**
   * Instead of treating Store IDs as continuous numbers (which introduces false hierarchy), each store is characterized by its historical sales volume computed prior to the test cutoff.
4. **Annual Seasonal Lag 52:**
   * Captures the exact sales from 1 year prior, representing **~58% of the model's total predictive power** due to recurring Thanksgiving and Christmas cycles.

---

## 🚀 Getting Started

### 1. Clone & Setup Environment
```bash
git clone https://github.com/adeeshachathurmina-ux/RetailPulse-AI.git
cd RetailPulse-AI

# Create virtual environment
python -m venv .venv

# Activate environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Automated Unit Tests
```bash
python -m unittest discover -s tests
```

### 3. Train & Evaluate Models
```bash
python train_model.py
```

### 4. Launch Streamlit Web Application
```bash
streamlit run app.py
```

---

## 🐳 Docker Deployment
To run with Docker:
```bash
docker build -t retailpulse-ai .
docker run -p 8501:8501 retailpulse-ai
```
Then navigate to `http://localhost:8501` in your browser.

---

## 📂 Repository Architecture
```text
RetailPulse-AI/
├── app.py                     # Interactive Streamlit Web Application
├── train_model.py             # Machine Learning pipeline & cross-validation
├── requirements.txt           # Production dependencies
├── Dockerfile                 # Containerization specification
├── .gitignore                 # Version control hygiene
├── data/
│   └── Walmart_Store_sales.csv# 6,435 weekly observations (45 stores)
├── models/
│   ├── best_model.joblib      # Serialized champion model
│   ├── metadata.json          # Metrics, CV scores, and feature importance
│   └── test_predictions.csv   # Holdout test set predictions
├── notebooks/                 # Academic exploration & proof-of-concept
│   ├── 01_Data_Understanding.ipynb
│   ├── 02_Time_Series_EDA.ipynb
│   ├── 03_Modelling_and_Evaluation.ipynb
│   └── 04_SARIMAX_Store_Example.ipynb
└── tests/
    └── test_pipeline.py       # Unit tests for data loading & leakage checks
```

---

## 🏆 Model Performance Benchmark

| Model Architecture | Holdout MAE | Holdout RMSE | MAPE | R² Score | Key Strength |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Naive Baseline (Lag 1)** | $51,838 | $79,029 | 4.99% | 0.9778 | Simple last-week benchmark |
| **Seasonal Naive (Lag 52)** | $53,465 | $84,106 | 5.46% | 0.9748 | Yearly repeat benchmark |
| **Random Forest Regressor** | $39,443 | $64,501 | 3.89% | 0.9852 | Robust non-linear interactions |
| **🏆 Gradient Boosting (Champion)** | **$41,125** | **$64,237** | **4.07%** | **0.9853** | Lowest RMSE & highest generalization |

---

## ⚠️ Limitations & Notes
* The historical dataset spans from **February 2010 to October 2012**. Macroeconomic inputs for the future simulator are user scenario assumptions rather than live macroeconomic feeds.
* In multi-step recursive forecasting ($h > 1$), prediction error compounds with the horizon. The application transparently visualizes this using expanding prediction intervals.

---

## 👨‍💻 Author
**RetailPulse AI Project**  
*BSc (Hons) in Data Science Undergraduate Portfolio Project*
