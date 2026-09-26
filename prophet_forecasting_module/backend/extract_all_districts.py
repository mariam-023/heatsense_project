"""
Batch District GEE LST Time-Series Extractor
=============================================
Batch extracts authentic historical MODIS LST observations for all 31 Karnataka districts
from Google Earth Engine (FAO GAUL Level 2 collection).

Usage:
  python -m prophet_forecasting_module.backend.extract_all_districts
"""

import sys
import sqlite3
from pathlib import Path
import pandas as pd

# Path setup
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from prophet_forecasting_module.backend.gee_extractor import extract_district_lst_series, initialize_gee_client
from prophet_forecasting_module.backend.prophet_model import load_and_preprocess_historical_data, run_prophet_forecast

DB_PATH = BASE_DIR / "instance" / "heatsense.sqlite"
if not DB_PATH.exists():
    DB_PATH = BASE_DIR / "app" / "database.sqlite"


def run_batch_district_extraction(force_refresh=False):
    print("==========================================================")
    print("STARTING BATCH GEE LST EXTRACTION FOR ALL KARNATAKA DISTRICTS")
    print("==========================================================")

    initialize_gee_client()

    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database file not found at: {DB_PATH}")

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    districts = conn.execute("SELECT id, name, latitude, longitude FROM districts ORDER BY id").fetchall()
    conn.close()

    print(f"Loaded {len(districts)} districts from database.\n")

    summary = []

    for d in districts:
        d_id = d['id']
        d_name = d['name']
        lat = d['latitude']
        lon = d['longitude']

        print(f"--- Processing District {d_id}: {d_name} ({lat:.4f}, {lon:.4f}) ---")
        try:
            df_hist = extract_district_lst_series(
                district_id=d_id,
                district_name=d_name,
                lat=lat,
                lon=lon,
                force_refresh=force_refresh
            )

            # Fit Prophet model
            df_clean = load_and_preprocess_historical_data(district_id=f"district_{d_id}")
            forecast_fut, _ = run_prophet_forecast(df_clean, location_key=f"district_{d_id}")

            summary.append({
                'district_id': d_id,
                'district_name': d_name,
                'historical_records': len(df_hist),
                'forecast_records': len(forecast_fut),
                'mean_raw_lst': round(float(df_hist['LST'].mean()), 2),
                'status': 'SUCCESS'
            })
        except Exception as err:
            print(f"[Batch Warning] Failed for district {d_id} ({d_name}): {err}")
            summary.append({
                'district_id': d_id,
                'district_name': d_name,
                'historical_records': 0,
                'forecast_records': 0,
                'mean_raw_lst': None,
                'status': f'FAILED: {err}'
            })

    print("\n==========================================================")
    print("BATCH EXTRACTION & MODEL TRAINING SUMMARY")
    print("==========================================================")
    df_summary = pd.DataFrame(summary)
    print(df_summary.to_string(index=False))

    return df_summary


if __name__ == "__main__":
    run_batch_district_extraction()
