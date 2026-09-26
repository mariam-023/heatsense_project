import sys
import os
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app import create_app

def run_location_sync_test():
    print("=================================================================")
    print("STARTING LOCATION SYNC & PROPHET FORECAST INTEGRATION TEST")
    print("=================================================================\n")

    app = create_app()
    client = app.test_client()

    # Test 1: District ID 1 (Bengaluru Urban)
    print("--- TEST 1: Request by district_id=1 (Bengaluru Urban) ---")
    res1 = client.get('/api/forecast?district_id=1')
    assert res1.status_code == 200, f"Expected status 200, got {res1.status_code}"
    data1 = res1.get_json()
    print(f"Status: {data1['status']}")
    print(f"Location Label: {data1['location_label']}")
    print(f"Data Scope: {data1['data_scope']}")
    print(f"Spatial Unit: {data1['spatial_unit']}")
    assert data1['data_scope'] == 'district_extracted'
    assert data1['location_label'] == 'Bengaluru Urban'
    assert data1['spatial_unit'] == 'District GAUL Polygon'

    # Test 2: Coordinates matching Bengaluru Urban
    print("\n--- TEST 2: Request by lat/lon/name (12.9716, 77.5946, Bengaluru Urban) ---")
    res2 = client.get('/api/forecast?lat=12.9716&lon=77.5946&name=Bengaluru%20Urban')
    assert res2.status_code == 200
    data2 = res2.get_json()
    print(f"Location Label: {data2['location_label']}")
    print(f"Data Scope: {data2['data_scope']}")
    assert data2['location_label'] == 'Bengaluru Urban'

    # Test 3: District ID 6 (Kalaburagi)
    print("\n--- TEST 3: Request by district_id=6 (Kalaburagi) ---")
    res3 = client.get('/api/forecast?district_id=6')
    assert res3.status_code == 200
    data3 = res3.get_json()
    print(f"Location Label: {data3['location_label']}")
    print(f"Data Scope: {data3['data_scope']}")
    assert data3['data_scope'] == 'district_extracted'
    assert data3['location_label'] == 'Kalaburagi'

    # Test 4: Regional Baseline Toggle
    print("\n--- TEST 4: Request Regional Baseline (baseline=true) ---")
    res4 = client.get('/api/forecast?baseline=true')
    assert res4.status_code == 200
    data4 = res4.get_json()
    print(f"Location Label: {data4['location_label']}")
    print(f"Data Scope: {data4['data_scope']}")
    assert data4['data_scope'] == 'regional_statewide'
    assert 'Statewide' in data4['location_label']

    # Test 5: Verify independence of forecast predictions
    fc1_temp = data1['forecast'][0]['yhat']
    fc3_temp = data3['forecast'][0]['yhat']
    fc4_temp = data4['forecast'][0]['yhat']
    print(f"\nBengaluru Urban Forecast Step 0: {fc1_temp}°C")
    print(f"Kalaburagi Forecast Step 0:      {fc3_temp}°C")
    print(f"Statewide Baseline Step 0:      {fc4_temp}°C")
    assert fc1_temp != fc3_temp, "District forecasts must update independently!"

    print("\n=================================================================")
    print("ALL LOCATION SYNC & PROPHET FORECAST TESTS PASSED SUCCESSFULLY!")
    print("=================================================================")

if __name__ == "__main__":
    run_location_sync_test()
