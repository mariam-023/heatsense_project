"""
Prophet Forecasting Engine Module
=================================
Self-contained Prophet time-series model training, fitting, and prediction script.

Features:
- Historical MODIS LST unit conversion (Kelvin -> Celsius)
- MODIS Terra orbital drift correction (+0.21°C/year since 2010)
- Prophet model fitting with yearly seasonality and linear trend
- 10-year daily forecast generation (3,650 periods)
- Temperature clipping (15.0°C to 50.0°C) to prevent unrealistic long-term drift
- Location/district specific prediction export (prediction_<location_key>.csv)
"""

import os
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import pandas as pd
import numpy as np
from prophet import Prophet

# Base directory for module relative paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

INPUT_TS_FILE = DATA_DIR / "Karnataka_LST_TimeSeries_2010_2026.csv"
FALLBACK_TS_FILE = DATA_DIR / "Karnataka_LST_TimeSeries.csv"
OUTPUT_PREDICTION_FILE = DATA_DIR / "prediction.csv"


def get_historical_file_for_location(location_key='regional'):
    """
    Returns the Path to the historical time series file for a given location key or district ID.
    For 'regional', defaults to Karnataka_LST_TimeSeries_2010_2026.csv.
    For specific location/district keys:
      - Checks data/historical_district_<id>.csv if integer or district_<id>
      - Checks data/historical_<loc_slug>.csv
    """
    if location_key in (None, '', 'regional', 'statewide', 'karnataka'):
        return INPUT_TS_FILE if INPUT_TS_FILE.exists() else FALLBACK_TS_FILE
    
    loc_str = str(location_key).strip().lower()

    # Check numeric district_id
    if loc_str.isdigit():
        dist_file = DATA_DIR / f"historical_district_{loc_str}.csv"
        if dist_file.exists():
            return dist_file
        
    if loc_str.startswith('district_'):
        dist_file = DATA_DIR / f"historical_{loc_str}.csv"
        if dist_file.exists():
            return dist_file

    loc_slug = loc_str.replace(' ', '_').replace('-', '_')
    custom_file = DATA_DIR / f"historical_{loc_slug}.csv"
    if custom_file.exists():
        return custom_file
    
    # Check if there is a historical_district_<id>.csv matching by slug name (e.g. bengaluru_urban)
    return None


def get_prediction_file_for_location(location_key='regional'):
    """
    Returns the Path to the output prediction file for a given location key or district ID.
    """
    if location_key in (None, '', 'regional', 'statewide', 'karnataka'):
        return OUTPUT_PREDICTION_FILE
    
    loc_str = str(location_key).strip().lower()
    if loc_str.isdigit():
        return DATA_DIR / f"prediction_district_{loc_str}.csv"
    if loc_str.startswith('district_'):
        return DATA_DIR / f"prediction_{loc_str}.csv"
        
    loc_slug = loc_str.replace(' ', '_').replace('-', '_')
    return DATA_DIR / f"prediction_{loc_slug}.csv"


def load_and_preprocess_historical_data(file_path=None, location_key='regional'):
    """
    Load historical time series dataset and apply MODIS unit conversions
    and orbital drift corrections.
    """
    if file_path is None:
        file_path = get_historical_file_for_location(location_key)

    if file_path is None or not Path(file_path).exists():
        raise FileNotFoundError(f"No authentic historical LST observation file found for location/district '{location_key}'")

    print(f"Loading historical LST time series from: {file_path}")
    df = pd.read_csv(file_path)

    # Clean missing values
    df = df.dropna(subset=['LST']).copy()

    # Convert MODIS LST (expressed in Kelvin * 50) to Celsius (°C)
    df['y'] = df['LST'] * 0.02 - 273.15
    df['ds'] = pd.to_datetime(df['date'])

    # Orbital drift correction: MODIS Terra orbital drift correction (+0.21°C/year since 2010)
    df['y'] = df['y'] + 0.21 * (df['ds'].dt.year - 2010)

    df_clean = df[['ds', 'y']].sort_values('ds').reset_index(drop=True)
    return df_clean


def run_prophet_forecast(df_historical, periods=3650, freq='D', output_file=None, location_key='regional'):
    """
    Fit Prophet model on historical series and generate future predictions.
    """
    if output_file is None:
        output_file = get_prediction_file_for_location(location_key)

    print(f"Training Prophet time-series model for location '{location_key}'...")
    # Fit linear trend with yearly seasonality and no changepoints to avoid extrapolating local noise
    model = Prophet(yearly_seasonality=True, changepoints=[])
    model.fit(df_historical)

    print(f"Generating {periods}-step ({freq}) future dataframe...")
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)

    # Clip extreme predictions to physically plausible bounds for surface temperature
    CLIP_MIN = 15.0
    CLIP_MAX = 50.0
    for col in ['yhat', 'yhat_lower', 'yhat_upper']:
        if col in forecast.columns:
            forecast[col] = forecast[col].clip(lower=CLIP_MIN, upper=CLIP_MAX)

    # Separate future predictions from historical baseline window
    historical_end_date = df_historical['ds'].max()
    forecast_future = forecast[forecast['ds'] > historical_end_date].copy()

    # Select output columns
    out_cols = ['ds', 'yhat']
    if 'yhat_lower' in forecast_future.columns and 'yhat_upper' in forecast_future.columns:
        out_cols.extend(['yhat_lower', 'yhat_upper'])

    # Ensure output directory exists and write atomically using temp file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_file = output_path.with_suffix('.tmp')
    forecast_future[out_cols].to_csv(temp_file, index=False)
    if temp_file.exists():
        if output_path.exists():
            try:
                output_path.unlink()
            except Exception:
                pass
        try:
            temp_file.replace(output_path)
        except Exception:
            forecast_future[out_cols].to_csv(output_path, index=False)
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except Exception:
                    pass
    print(f"Exported {len(forecast_future)} future prediction rows to: {output_path}")

    return forecast_future, forecast


if __name__ == "__main__":
    print("=== Standalone Prophet Forecasting Execution ===")
    df_hist = load_and_preprocess_historical_data()
    forecast_future, full_forecast = run_prophet_forecast(df_hist)
    print(f"Historical range: {df_hist['ds'].min().date()} to {df_hist['ds'].max().date()}")
    print(f"Future forecast range: {forecast_future['ds'].min().date()} to {forecast_future['ds'].max().date()}")
