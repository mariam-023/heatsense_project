# 📈 Standalone Prophet Forecasting Module

A self-contained, portable time-series prediction package extracted from the HeatSense project. This package provides long-term Land Surface Temperature (LST) forecasting using Meta Prophet, complete with a REST API backend, preprocessed historical MODIS data, prediction generator, and an interactive Chart.js frontend dashboard.

---

## 📁 Package Directory Structure

```
prophet_forecasting_module/
├── README.md                              # Comprehensive guide & integration documentation
├── requirements.txt                       # Python package dependencies
├── .env.example                           # Environment configuration template
├── data/
│   ├── Karnataka_LST_TimeSeries_2010_2026.csv # Input historical satellite LST dataset
│   └── prediction.csv                     # Formatted 10-year daily forecast output CSV
├── backend/
│   ├── prophet_model.py                   # Self-contained Prophet fitting & prediction engine
│   └── app.py                             # Flask REST API server & static file host
└── frontend/
    ├── index.html                         # Standalone HTML visualization dashboard
    ├── styles.css                         # Responsive dashboard styling
    └── app.js                             # API client & Chart.js rendering logic
```

---

## 🛠️ Prerequisites & Installation

### Requirements
- Python 3.9+ (Python 3.10+ recommended)
- `pip` package manager

### Setup Steps
1. Navigate to the `prophet_forecasting_module/` directory:
   ```bash
   cd prophet_forecasting_module
   ```
2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```
3. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage Guide

### 1. Re-Run Model Training & Prediction Generation
To fit the Prophet model on the historical time-series data and export fresh predictions to `data/prediction.csv`:
```bash
python backend/prophet_model.py
```

### 2. Start the Backend API & Web Dashboard
Launch the standalone Flask server:
```bash
python backend/app.py
```
The server will start at **`http://127.0.0.1:5000`**.

- Access the web dashboard in your browser: **`http://127.0.0.1:5000/`**
- Check system status: **`http://127.0.0.1:5000/api/health`**
- Fetch JSON forecast data: **`http://127.0.0.1:5000/api/forecast`**

---

## 🔌 API Endpoint Reference

### 1. `GET /api/health`
Returns system status and verifies data file existence.
```json
{
  "status": "healthy",
  "service": "Prophet Forecasting Engine",
  "input_dataset_exists": true,
  "input_dataset_path": "Karnataka_LST_TimeSeries_2010_2026.csv",
  "output_prediction_exists": true,
  "output_prediction_path": "prediction.csv"
}
```

### 2. `GET /api/forecast`
Returns future predictions and historical baseline records.
- **Query Parameters:**
  - `format`: `json` (default) or `csv`.
  - `include_historical`: `true` (default) or `false`.
- **Sample JSON Response:**
```json
{
  "status": "success",
  "historical_count": 768,
  "forecast_count": 3650,
  "historical": [
    { "ds": "2010-01-01", "temp": 28.45 }
  ],
  "forecast": [
    { "ds": "2026-09-07", "yhat": 31.25, "yhat_lower": 29.80, "yhat_upper": 32.70 }
  ]
}
```

### 3. `POST /api/forecast/generate`
Triggers on-demand retraining of the Prophet model and exports updated predictions.
- **Query Parameters:**
  - `periods`: integer forecast steps (default: `3650` days).

---

## 🧩 How to Integrate Into Another Project

### Embedding in a Third-Party Frontend
1. Copy `frontend/app.js`, `frontend/index.html` (or chart component), and `frontend/styles.css` into your web application.
2. In your HTML, define the backend URL before loading `app.js`:
   ```html
   <script>
     window.PROPHET_API_BASE_URL = "http://your-backend-domain.com";
   </script>
   <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
   <script src="app.js"></script>
   ```

### Importing into Another Python Backend
You can import the forecasting engine directly into another Python application:
```python
from prophet_forecasting_module.backend.prophet_model import (
    load_and_preprocess_historical_data,
    run_prophet_forecast
)

# Load data and run 5-year forecast
df_hist = load_and_preprocess_historical_data("path/to/your_timeseries.csv")
future_df, full_forecast = run_prophet_forecast(df_hist, periods=1825, freq='D')
```

---

## 🔬 Model Specification & Methodological Details

- **Input Scaling:** Converts raw MODIS LST values into Celsius: $y = \text{LST} \times 0.02 - 273.15$.
- **Orbital Drift Correction:** Adds $+0.21^\circ\text{C}/\text{year}$ starting from 2010 to correct for MODIS Terra satellite overpass time drift.
- **Prophet Configuration:** `Prophet(yearly_seasonality=True, changepoints=[])` fits a stable linear trend with annual seasonality without over-extrapolating short-term noise.
- **Temperature Clipping:** Predictions are constrained to $[15.0^\circ\text{C}, 50.0^\circ\text{C}]$.

---

## 📋 License & Notes
Extracted as a standalone module. Preserves original project forecasting logic and dataset schema without external dependencies on Random Forest or GEE components.
