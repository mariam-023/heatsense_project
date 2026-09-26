import os
import sys
from pathlib import Path
import pandas as pd

# Add app directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app import create_app
from prophet_forecasting_module.backend.prophet_model import DATA_DIR, run_prophet_forecast, load_and_preprocess_historical_data

def run_tests():
    app = create_app()
    client = app.test_client()

    print("=== TEST 1: Regional Statewide Forecast ===")
    res_reg = client.get('/api/forecast?location=regional')
    assert res_reg.status_code == 200, f"Expected 200, got {res_reg.status_code}"
    data_reg = res_reg.get_json()
    print(f"Status: {data_reg['status']}")
    print(f"Scope: {data_reg['data_scope']}")
    print(f"Label: {data_reg['location_label']}")
    print(f"Historical Count: {data_reg['historical_count']}, Forecast Count: {data_reg['forecast_count']}")
    assert data_reg['data_scope'] == 'regional_statewide'
    assert 'Karnataka Statewide' in data_reg['location_label']
    print("Test 1 Passed!\n")

    print("=== TEST 2: Unextracted Location (Fallback Notice) ===")
    res_unextracted = client.get('/api/forecast?location=unextracted_city')
    assert res_unextracted.status_code == 200, f"Expected 200, got {res_unextracted.status_code}"
    data_unex = res_unextracted.get_json()
    print(f"Scope: {data_unex['data_scope']}")
    print(f"Has Local Data: {data_unex['has_local_data']}")
    print(f"Notice: {data_unex['notice']}")
    assert data_unex['data_scope'] == 'regional_fallback'
    assert data_unex['has_local_data'] is False
    assert 'GEE authentication required' in data_unex['notice']
    print("Test 2 Passed!\n")

    print("=== TEST 3: Genuine Multi-Location Extraction Simulation & Independent Prophet Fits ===")
    # Load base regional structure for dates
    df_base = pd.read_csv(DATA_DIR / "Karnataka_LST_TimeSeries_2010_2026.csv")

    # Create Location 1 (Bengaluru Urban: cooler baseline ~15400 MODIS raw units)
    df_bengaluru = df_base.copy()
    df_bengaluru['LST'] = 15350.0 + (df_bengaluru.index % 46) * 12.0

    # Create Location 2 (Kalaburagi: hotter northern dryland baseline ~15900 MODIS raw units)
    df_kalaburagi = df_base.copy()
    df_kalaburagi['LST'] = 15850.0 + (df_kalaburagi.index % 46) * 18.0

    file_bengaluru = DATA_DIR / "historical_bengaluru_urban.csv"
    file_kalaburagi = DATA_DIR / "historical_kalaburagi.csv"

    df_bengaluru.to_csv(file_bengaluru, index=False)
    df_kalaburagi.to_csv(file_kalaburagi, index=False)

    print(f"Saved test historical CSV for Bengaluru Urban to: {file_bengaluru}")
    print(f"Saved test historical CSV for Kalaburagi to: {file_kalaburagi}")

    # Query API for Bengaluru Urban
    res_ben = client.get('/api/forecast?location=bengaluru_urban')
    assert res_ben.status_code == 200
    data_ben = res_ben.get_json()

    # Query API for Kalaburagi
    res_kal = client.get('/api/forecast?location=kalaburagi')
    assert res_kal.status_code == 200
    data_kal = res_kal.get_json()

    print(f"Bengaluru Scope: {data_ben['data_scope']}, Label: {data_ben['location_label']}")
    print(f"Kalaburagi Scope: {data_kal['data_scope']}, Label: {data_kal['location_label']}")

    assert data_ben['data_scope'] == 'local_extracted'
    assert data_kal['data_scope'] == 'local_extracted'

    hist_ben_temp0 = data_ben['historical'][0]['temp']
    hist_kal_temp0 = data_kal['historical'][0]['temp']
    fc_ben_yhat0 = data_ben['forecast'][0]['yhat']
    fc_kal_yhat0 = data_kal['forecast'][0]['yhat']

    print(f"Bengaluru Historical[0]: {hist_ben_temp0}°C | Forecast[0]: {fc_ben_yhat0}°C")
    print(f"Kalaburagi Historical[0]: {hist_kal_temp0}°C | Forecast[0]: {fc_kal_yhat0}°C")

    # Verify that temperatures are genuinely different and independent!
    assert hist_ben_temp0 != hist_kal_temp0, "Historical temperatures should differ!"
    assert fc_ben_yhat0 != fc_kal_yhat0, "Forecast predictions should differ!"

    print("Test 3 Passed! Independent multi-location forecasts verified successfully.\n")
    print("ALL TESTS PASSED CLEANLY!")

if __name__ == "__main__":
    run_tests()
