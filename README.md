# Weather Time Series Forecasting with Deep Learning
 
![License](https://img.shields.io/badge/license-GNU%20GPL-green.svg)

This repository contains the code and resources developed for the **Final Degree Project (TFG)**:  
**“Predicción de series temporales meteorológicas mediante técnicas de aprendizaje profundo”**  
(*Weather time series forecasting using deep learning techniques*)  
by **José Ramón Morera Campos**, Universidad de La Laguna, 2025.

The published report is available [here](https://riull.ull.es/xmlui/handle/915/43055).

---

## 📌 Project Overview

The goal of this project is to develop a **short-term weather forecasting system** (3-12h) using **Deep Learning**.  
The models are trained on meteorological data from multiple weather stations in **Tenerife (Canary Islands, Spain)**, and evaluated for their ability to **generalize to unseen locations without retraining** (zero-shot setting).

The system integrates:
- **Data acquisition pipelines** (GRAFCAN and Open-Meteo APIs).
- **Data preprocessing** (missing value imputation, normalization, anomaly detection, temporal encoding).  
- **Forecasting models** (ARIMA, LSTM, CNN, and CNN-LSTM hybrid).  
- **Deployment** as a web application for real-time forecasting.

## Contents 
This repository contains the core forecasting models, preprocessing code, and deployment components. The acquisition pipelines are not included at the moment due to API key usage and deployment environment coupling (TODO).

---

## 🛠️ Tech Stack

- ![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg) for data processing and ML models  
- ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg) + Keras for deep learning  
- ![FastAPI](https://img.shields.io/badge/FastAPI-backend-teal.svg) for serving predictions via REST API  
- ![Docker](https://img.shields.io/badge/Docker-containerization-blue.svg) for deployment  
- ![TimescaleDB](https://img.shields.io/badge/TimescaleDB-time%20series%20DB-purple.svg) for scalable time series storage  
- ![Node-RED](https://img.shields.io/badge/Node--RED-data%20pipelines-red.svg) for data acquisition orchestration  

---

## 🧪 Methodology

1. **Data Acquisition**
   - GRAFCAN (official Canary Islands weather stations).  
   - Open-Meteo (ICON Global, ARPEGE Europe models).  
   - Uses NodeRed to schedule data acquisition through REST APIs.
   - Stored in a **TimescaleDB** database (PostgreSQL extension for time series).

2. **Preprocessing**
   - Missing data imputation (PCHIP & day-lag methods).  
   - Outlier detection (KNN distance method).  
   - Fourier analysis for periodicity (daily & yearly cycles).  
   - Temporal encoding (sine/cosine for day/year cycles).  
   - Sliding window dataset generation.

3. **Models**
   - **ARIMA** (baseline).  
   - **CNN** (1D convolutions for local temporal patterns).  
   - **LSTM** (long-term dependencies).  
   - **Hybrid CNN-LSTM** (combines local + sequential learning).  
   - Training with **TensorFlow/Keras**, hyperparameter tuning via **Keras Tuner**.

4. **Deployment**
   - **Backend**: Python + FastAPI + Celery.  
   - **Frontend**: Next.js + React. Web interface for loading live station data and visualizing forecasts.  
   - Containerized with **Docker** for reproducibility.

---

## 📊 Results

- **LSTM models** achieved the best accuracy in short-term predictions (3–12h).  
- Outperformed classical methods (ARIMA).  
- Improved over existing solutions by reducing error 50%.
- The system demonstrated the ability to forecast **unseen locations (zero-shot)** with good reliability.  
- Final application is modular and scalable for real-world use.

---

## Data & Reproducibility
Currently the data used is not available due to its volume.

However, using the acquisition pipeline provided (TODO), and the relevant free API keys (GRAFCAN & Open-Meteo), it should be easy to retrieve using Node-RED.

The trained model weights are currently not uploaded due to the size constraints, however, training from scratch should be easy using the provided notebooks and the parameter values specified in the report.

## 🚀 Getting Started

### Requirements
- Python 3.10+
- [TensorFlow](https://www.tensorflow.org/)
- [Keras](https://keras.io/)
- [Pandas](https://pandas.pydata.org/)
- [TimescaleDB](https://www.timescale.com/) (for data storage)
- [Docker](https://www.docker.com/) (optional, for deployment)

### Launch Web Application

    uvicorn app.main:app --reload

Access the frontend at:  
➡️ `http://127.0.0.1:8000`

---


## 📂 Repository Structure
```
├── app/                        # Full-stack application for model deployment
│   ├── backend_api/            # REST API (FastAPI/Flask) to serve forecasts
│   │   └── debug_inputs/       # Sample JSON payloads for testing API endpoints
│   ├── frontend_inference/     # Web dashboard (React/Next.js) for visualizing results
│   │   ├── app/                # Main application routing and pages
│   │   ├── components/         # Reusable UI elements
│   │   ├── lib/                # Frontend utility functions
│   │   └── public/             # Static assets (icons, images)
│   └── locust/                 # Load testing scripts to stress-test the API
├── data/                       # Dataset storage (typically .gitignored)
│   ├── raw/                    # Original, immutable time series data
│   └── windows/                # Processed data split into sliding windows (Train/Val)
├── notebooks/                  # Jupyter notebooks for experimentation & analysis
│   ├── eda/                    # Exploratory Data Analysis & visual inspection
│   └── eval/                   # Model performance evaluation & comparisons
├── results/                    # Generated artifacts, logs, and metrics
│   ├── ARIMA/                  # Forecast outputs/parameters for statistical models
│   ├── CNN/                    # Training logs for Convolutional models
│   ├── windows_gridsearch/     # Hyperparameter optimization logs
│   └── windows_results/        # Final performance metrics on test sets
├── src/                        # Core Python source code (The "Library")
│   ├── data/                   # Scripts for preprocessing and window generation
│   └── models/                 # Model architecture definitions
│       ├── ARIMA/              # Statistical model logic
│       ├── CNN/                # Deep Learning architecture (1D-CNN)
│       └── LSTM/               # Recurrent Neural Network architecture
└── TFG_Report_Latex/           # Bachelor's Thesis document source
    └── images/                 # Plots and diagrams generated for the report
    ```


## ToDo
1) Upload to the repo the data ingestion pipeline.
2) Restructure using MLFlow or similar for clarity.

