import sys
import os
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import pandas as pd

# Path setup
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prophet_forecasting_module.backend.gee_extractor import extract_district_lst_series
from app import create_app

def run_district_verification():
    print("=================================================================")
    print("STARTING GEE DISTRICT EXTRACTION & PROPHET VERIFICATION TEST")
    print("=================================================================\n")

    # 1. Extract Bengaluru Urban (District 1)
    print("--- STEP 1: Extracting GEE LST Time Series for Bengaluru Urban (District 1) ---")
    df_bengaluru = extract_district_lst_series(
        district_id=1,
        district_name="Bengaluru Urban",
        lat=12.9716,
        lon=77.5946,
        force_refresh=True
    )
    print(f"Bengaluru Urban shape: {df_bengaluru.shape}")
    print(f"Columns: {list(df_bengaluru.columns)}")
    print(f"Sample head:\n{df_bengaluru[['district_id', 'district_name', 'date', 'LST', 'valid_pixel_count']].head(3)}\n")

    # 2. Extract Kalaburagi (District 6)
    print("--- STEP 2: Extracting GEE LST Time Series for Kalaburagi (District 6) ---")
    df_kalaburagi = extract_district_lst_series(
        district_id=6,
        district_name="Kalaburagi",
        lat=17.3297,
        lon=76.8343,
        force_refresh=True
    )
    print(f"Kalaburagi shape: {df_kalaburagi.shape}")
    print(f"Columns: {list(df_kalaburagi.columns)}")
    print(f"Sample head:\n{df_kalaburagi[['district_id', 'district_name', 'date', 'LST', 'valid_pixel_count']].head(3)}\n")

    # 3. Verify datasets are distinct and valid
    assert not df_bengaluru.empty, "Bengaluru Urban DataFrame is empty!"
    assert not df_kalaburagi.empty, "Kalaburagi DataFrame is empty!"
    assert df_bengaluru['district_id'].iloc[0] == 1
    assert df_kalaburagi['district_id'].iloc[0] == 6

    lst_ben_avg = df_bengaluru['LST'].mean()
    lst_kal_avg = df_kalaburagi['LST'].mean()

    print(f"Bengaluru Urban Mean Raw LST: {lst_ben_avg:.2f}")
    print(f"Kalaburagi Mean Raw LST:       {lst_kal_avg:.2f}")
    assert lst_ben_avg != lst_kal_avg, "District mean LST values should not be identical!"
    print("District LST observations verified as authentic and distinct!\n")

    # 4. Test Flask API Endpoints for District Forecasts
    print("--- STEP 3: Testing Flask API Endpoints for Per-District Forecasts ---")
    app = create_app()
    client = app.test_client()

    res_ben = client.get('/api/forecast?district_id=1')
    assert res_ben.status_code == 200
    data_ben = res_ben.get_json()
    print(f"Bengaluru API Scope: {data_ben['data_scope']}, Label: {data_ben['location_label']}")

    res_kal = client.get('/api/forecast?district_id=6')
    assert res_kal.status_code == 200
    data_kal = res_kal.get_json()
    print(f"Kalaburagi API Scope: {data_kal['data_scope']}, Label: {data_kal['location_label']}")

    assert data_ben['data_scope'] == 'district_extracted'
    assert data_kal['data_scope'] == 'district_extracted'
    assert data_ben['location_label'] == 'Bengaluru Urban'
    assert data_kal['location_label'] == 'Kalaburagi'

    fc_ben_0 = data_ben['forecast'][0]['yhat']
    fc_kal_0 = data_kal['forecast'][0]['yhat']
    print(f"Bengaluru Urban Forecast[0]: {fc_ben_0}°C")
    print(f"Kalaburagi Forecast[0]:       {fc_kal_0}°C")
    assert fc_ben_0 != fc_kal_0, "District forecasts must update independently!"

    print("\n=================================================================")
    print("ALL DISTRICT GEE EXTRACTION AND PROPHET VERIFICATION TESTS PASSED!")
    print("=================================================================")

if __name__ == "__main__":
    run_district_verification()
