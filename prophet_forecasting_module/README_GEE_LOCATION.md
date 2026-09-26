# Google Earth Engine (GEE) Location-Specific Time-Series Guide

This guide documents the Google Earth Engine (GEE) extraction pipeline, dependencies, permissions, and workflow for producing authentic location-specific Land Surface Temperature (LST) historical observations for Meta Prophet model training in HeatSense.

---

## 1. GEE Infrastructure & Prerequisites

### Required Python Libraries
Ensure the following packages are installed in your environment:
```bash
pip install earthengine-api geemap pandas prophet
```

### Authentication & Project Setup
1. **Authenticate GEE Credentials**:
   ```bash
   earthengine authenticate
   ```
   Follow the web browser prompt to log into your Google Account registered with Google Earth Engine.

2. **Configure Cloud Project ID**:
   Set your Google Cloud Project ID as an environment variable before running HeatSense or extraction scripts:
   ```cmd
   set EE_PROJECT=your-google-cloud-project-id
   ```
   or in PowerShell:
   ```powershell
   $env:EE_PROJECT="your-google-cloud-project-id"
   ```

---

## 2. Satellite Datasets & Preprocessing

The extraction pipeline consumes the following Earth Engine assets:
- **MODIS MOD11A2 Version 6.1** (`MODIS/061/MOD11A2`):
  - **Resolution**: 1km spatial, 8-day composite temporal resolution.
  - **Bands**: `LST_Day_1km`
  - **Scale Conversion**: $\text{LST (°C)} = (\text{LST\_Day\_1km} \times 0.02) - 273.15$
  - **Orbital Drift Correction**: $+0.21^\circ\text{C / year}$ applied from 2010 onward to correct for MODIS Terra overpass time drift.
- **Administrative Boundaries**:
  - `FAO/GAUL/2015/level2` (District boundaries for Karnataka, India).
  - Point buffer fallback ($10\text{ km}$ circular geometry around custom $\text{lat}/\text{lon}$ coordinates).

---

## 3. Running Location Historical Extraction

To extract an authentic historical observation dataset for a location, execute `gee_extractor.py`:

### Example A: Extracting by District Boundary (e.g. Bengaluru Urban)
```bash
python -m prophet_forecasting_module.backend.gee_extractor --location bengaluru_urban --district "Bangalore Urban"
```

### Example B: Extracting by City Coordinates (e.g. Kalaburagi)
```bash
python -m prophet_forecasting_module.backend.gee_extractor --location kalaburagi --lat 17.3297 --lon 76.8343
```

This exports `prophet_forecasting_module/data/historical_<location_slug>.csv`.

---

## 4. Prophet Model Training & Integration

Once `historical_<location_slug>.csv` exists in the `data/` directory:
1. The HeatSense backend automatically detects the authentic local observation dataset when `/api/forecast?location=<location_slug>` is called.
2. The Meta Prophet model fits on `historical_<location_slug>.csv` and exports predictions to `data/prediction_<location_slug>.csv`.
3. The UI scope badge updates to **`Scope: Local Satellite Extracted (<Location Name>)`** and displays genuine local historical and future projected temperatures.

---

## 5. Scope & Limitations

- **Statewide Regional Baseline**: If no extracted `historical_<location_slug>.csv` dataset exists for a requested city/location, HeatSense serves the **Karnataka Statewide (Regional Baseline)** and displays a notice banner stating that local satellite extraction is pending GEE authentication.
- **No Synthetic Data**: Statewide regional baseline data is never altered or passed off as local data.
- **Data Gaps**: High cloud cover during the Southwest Monsoon (June–September) may result in masked pixels; Prophet's linear trend with annual seasonality handles missing 8-day timesteps smoothly.
