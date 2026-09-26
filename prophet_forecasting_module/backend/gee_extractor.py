"""
Google Earth Engine (GEE) Satellite LST Time-Series Extractor
=============================================================
Extracts authentic dated Land Surface Temperature (LST) observations from MODIS (MOD11A2)
for Karnataka district boundary polygons (FAO GAUL Level 2) OR point-level GPS 1km MODIS pixels.

Saves structured CSV files to prophet_forecasting_module/data/:
  - historical_district_<id>.csv
  - historical_point_<lat2>_<lon2>.csv

Schema:
- location_key
- district_id / lat / lon
- district_name / location_name
- date
- LST
- dataset_source
- extraction_method
- valid_pixel_count
- .geo
"""

import os
import sys
import argparse
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import pandas as pd

# Path setup
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# Mapping from SQLite database district names to GEE FAO GAUL Level 2 ADM2_NAME strings
GAUL_NAME_MAP = {
    "Bengaluru Urban": "Bangalore Urban",
    "Bengaluru Rural": "Bangalore Rural",
    "Mysuru": "Mysore",
    "Tumakuru": "Tumkur",
    "Kolar": "Kolar",
    "Chikkaballapura": "Kolar",
    "Ramanagara": "Bangalore Rural",
    "Chamarajanagara": "Chamrajnagar",
    "Mandya": "Mandya",
    "Hassan": "Hassan",
    "Kodagu": "Kodagu",
    "Dakshina Kannada": "Dakshin Kannad",
    "Udupi": "Udupi",
    "Mangaluru": "Dakshin Kannad",
    "Shivamogga": "Shimoga",
    "Chikkamagaluru": "Chikmagalur",
    "Davanagere": "Davanagere",
    "Chitradurga": "Chitradurga",
    "Belagavi": "Belgaum",
    "Dharwad": "Dharwad",
    "Gadag": "Gadag",
    "Bagalkote": "Bagalkot",
    "Vijayapura": "Bijapur",
    "Hubli-Dharwad": "Dharwad",
    "Haveri": "Haveri",
    "Uttara Kannada": "Uttar Kannand",
    "Koppal": "Koppal",
    "Ballari": "Bellary",
    "Raichur": "Raichur",
    "Yadgir": "Gulbarga",
    "Kalaburagi": "Gulbarga",
    "Bidar": "Bidar",
}


def initialize_gee_client():
    """Initializes GEE using project 'heatsense' or environment variable."""
    import ee
    try:
        project = os.environ.get('EE_PROJECT') or os.environ.get('EARTHENGINE_PROJECT') or os.environ.get('GOOGLE_CLOUD_PROJECT') or 'heatsense'
        ee.Initialize(project=project)
        print(f"[GEE Extractor] Initialized GEE client with project: '{project}'")
        return True
    except Exception as e:
        print(f"[GEE Extractor Warning] ee.Initialize failed: {e}")
        try:
            ee.Initialize()
            print("[GEE Extractor] Initialized GEE client with default credentials.")
            return True
        except Exception as err:
            raise RuntimeError(f"Google Earth Engine connection failed: {err}")


def get_district_boundary_geometry(district_name, lat=None, lon=None, radius_meters=15000):
    """
    Retrieves official boundary polygon from FAO GAUL Level 2 collection for a district.
    Falls back to a circular geometry buffer around lat/lon if boundary polygon is not matched.
    """
    import ee

    gaul_name = GAUL_NAME_MAP.get(district_name, district_name)

    gaul_l2 = ee.FeatureCollection("FAO/GAUL/2015/level2")
    filtered = gaul_l2.filter(
        ee.Filter.And(
            ee.Filter.eq('ADM0_NAME', 'India'),
            ee.Filter.eq('ADM1_NAME', 'Karnataka'),
            ee.Filter.eq('ADM2_NAME', gaul_name)
        )
    )

    if filtered.size().getInfo() > 0:
        print(f"[GEE Extractor] Found official GAUL Level 2 boundary polygon for '{district_name}' (GAUL name: '{gaul_name}')")
        return filtered.geometry(), f"GAUL_L2_polygon_mean ({gaul_name})"

    if lat is not None and lon is not None:
        print(f"[GEE Extractor Warning] Boundary polygon for '{district_name}' not matched in GAUL. Using {radius_meters}m circular AOI buffer at ({lat}, {lon}).")
        return ee.Geometry.Point([lon, lat]).buffer(radius_meters), f"Circular_AOI_buffer_{radius_meters}m"

    raise ValueError(f"District geometry for '{district_name}' could not be resolved.")


def mask_modis_lst_image(image):
    """
    Applies MODIS/061/MOD11A2 quality control and valid land surface temperature range masking.

    MODIS MOD11A2 Product Specification & Tropical Land Range:
    - Valid LST Range for Land Surface Temperature: 13,657.5 to 16,907.5 (273.15 K / 0.0°C to 338.15 K / +65.0°C).
      Fill values / unmasked cloud-top shadows / open ocean pixels fall outside this range.
    - QC_Day mandatory quality flag: Bits 0-1 (QC_Day & 3) <= 1
      00: LST produced, good quality
      01: LST produced, reliable quality
    """
    lst = image.select('LST_Day_1km')
    qc = image.select('QC_Day')

    # 1. Physically valid land temperature range mask (0°C to 65°C)
    valid_range = lst.gte(13657.5).And(lst.lte(16907.5))

    # 2. QC_Day mandatory quality mask (bits 0-1 <= 1)
    good_quality = qc.bitwiseAnd(3).lte(1)

    clean_mask = valid_range.And(good_quality)
    return lst.updateMask(clean_mask)


def extract_district_lst_series(district_id, district_name, lat=None, lon=None, start_year=2010, end_year=2026, force_refresh=False):
    """
    Extracts authentic 8-day LST time series for a district polygon from MODIS (MOD11A2) via GEE.
    """
    import ee

    out_file = DATA_DIR / f"historical_district_{district_id}.csv"
    if out_file.exists() and not force_refresh:
        print(f"[GEE Extractor] Cached historical dataset exists for district {district_id} ({district_name}) at: {out_file}")
        return pd.read_csv(out_file)

    initialize_gee_client()

    geom, extraction_method = get_district_boundary_geometry(district_name, lat=lat, lon=lon)

    start_date = f"{start_year}-01-01"
    end_date = f"{end_year}-12-31"

    print(f"[GEE Extractor] Querying MODIS MOD11A2 Day 1km LST for District {district_id} ({district_name}) [{start_date} to {end_date}]...")

    modis_coll = (
        ee.ImageCollection('MODIS/061/MOD11A2')
        .filterBounds(geom)
        .filterDate(start_date, end_date)
        .select(['LST_Day_1km', 'QC_Day'])
    )

    total_scenes = modis_coll.size().getInfo()
    print(f"[GEE Extractor] Total MODIS timesteps found: {total_scenes}")

    def reduce_timestep(image):
        date_str = image.date().format('YYYY-MM-dd')
        sys_id = image.id()

        masked_lst = mask_modis_lst_image(image)

        mean_dict = masked_lst.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geom,
            scale=1000,
            maxPixels=1e9
        )

        count_dict = masked_lst.reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=geom,
            scale=1000,
            maxPixels=1e9
        )

        mean_lst = mean_dict.get('LST_Day_1km')
        valid_count = count_dict.get('LST_Day_1km')

        return ee.Feature(None, {
            'district_id': district_id,
            'district_name': district_name,
            'date': date_str,
            'LST': mean_lst,
            'dataset_source': 'MODIS/061/MOD11A2',
            'extraction_method': extraction_method,
            'valid_pixel_count': valid_count,
            'system_index': sys_id,
        })

    fc = modis_coll.map(reduce_timestep)
    features = fc.getInfo()['features']

    records = []
    for f in features:
        props = f['properties']
        raw_lst = props.get('LST')
        valid_pixels = int(props.get('valid_pixel_count') or 0)
        # Require valid LST observation and at least 5 valid pixels in district polygon
        if raw_lst is not None and not pd.isna(raw_lst) and valid_pixels >= 5:
            records.append({
                'district_id': props.get('district_id'),
                'district_name': props.get('district_name'),
                'date': props.get('date'),
                'LST': float(raw_lst),
                'dataset_source': props.get('dataset_source'),
                'extraction_method': props.get('extraction_method'),
                'valid_pixel_count': valid_pixels,
                'system:index': props.get('system_index'),
                '.geo': '{"type":"MultiPoint","coordinates":[]}'
            })

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError(f"No valid LST observations extracted for District {district_id} ({district_name}).")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False)
    print(f"[GEE Extractor] Successfully exported {len(df)} authentic historical LST observations for District {district_id} ({district_name}) to: {out_file}")
    return df


def extract_point_lst_series(lat, lon, location_name=None, start_year=2010, end_year=2026, buffer_meters=1000, force_refresh=False):
    """
    Extracts authentic 8-day LST time series for a GPS lat/lon coordinate from MODIS (MOD11A2) via GEE.
    Uses rounded 2-decimal grid cell key e.g. point_12.97_77.59 for caching.
    """
    import ee

    lat_key = f"{lat:.2f}"
    lon_key = f"{lon:.2f}"
    loc_key = f"point_{lat_key}_{lon_key}"

    out_file = DATA_DIR / f"historical_{loc_key}.csv"
    if out_file.exists() and not force_refresh:
        print(f"[GEE Extractor] Cached historical dataset exists for point ({lat:.4f}, {lon:.4f}) [grid {loc_key}] at: {out_file}")
        return pd.read_csv(out_file)

    initialize_gee_client()

    point_geom = ee.Geometry.Point([lon, lat]).buffer(buffer_meters)
    extraction_method = f"1km_MODIS_pixel_buffer ({lat_key}, {lon_key})"

    start_date = f"{start_year}-01-01"
    end_date = f"{end_year}-12-31"

    print(f"[GEE Extractor] Querying MODIS MOD11A2 1km Pixel for ({lat:.4f}, {lon:.4f}) [grid {loc_key}]...")

    modis_coll = (
        ee.ImageCollection('MODIS/061/MOD11A2')
        .filterBounds(point_geom)
        .filterDate(start_date, end_date)
        .select(['LST_Day_1km', 'QC_Day'])
    )

    total_scenes = modis_coll.size().getInfo()

    def reduce_point_timestep(image):
        date_str = image.date().format('YYYY-MM-dd')
        sys_id = image.id()

        masked_lst = mask_modis_lst_image(image)

        mean_dict = masked_lst.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point_geom,
            scale=1000,
            maxPixels=1e9
        )

        count_dict = masked_lst.reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=point_geom,
            scale=1000,
            maxPixels=1e9
        )

        return ee.Feature(None, {
            'location_key': loc_key,
            'lat': lat,
            'lon': lon,
            'location_name': location_name or f"{lat:.4f}, {lon:.4f}",
            'date': date_str,
            'LST': mean_dict.get('LST_Day_1km'),
            'dataset_source': 'MODIS/061/MOD11A2',
            'extraction_method': extraction_method,
            'valid_pixel_count': count_dict.get('LST_Day_1km'),
            'system_index': sys_id,
        })

    fc = modis_coll.map(reduce_point_timestep)
    features = fc.getInfo()['features']

    records = []
    for f in features:
        props = f['properties']
        raw_lst = props.get('LST')
        if raw_lst is not None and not pd.isna(raw_lst):
            records.append({
                'location_key': props.get('location_key'),
                'lat': props.get('lat'),
                'lon': props.get('lon'),
                'location_name': props.get('location_name'),
                'date': props.get('date'),
                'LST': float(raw_lst),
                'dataset_source': props.get('dataset_source'),
                'extraction_method': props.get('extraction_method'),
                'valid_pixel_count': int(props.get('valid_pixel_count') or 0),
                'system:index': props.get('system_index'),
                '.geo': '{"type":"MultiPoint","coordinates":[]}'
            })

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError(f"No valid LST observations extracted for Point ({lat}, {lon}).")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False)
    print(f"[GEE Extractor] Successfully exported {len(df)} authentic historical LST observations for Point ({lat:.4f}, {lon:.4f}) to: {out_file}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract location/district LST time series from GEE.")
    parser.add_argument("--district_id", type=int, help="District ID")
    parser.add_argument("--district_name", help="District Name")
    parser.add_argument("--lat", type=float, help="Latitude")
    parser.add_argument("--lon", type=float, help="Longitude")
    parser.add_argument("--start_year", type=int, default=2010)
    parser.add_argument("--end_year", type=int, default=2026)
    parser.add_argument("--refresh", action="store_true")

    args = parser.parse_args()
    try:
        if args.district_id and args.district_name:
            extract_district_lst_series(
                district_id=args.district_id,
                district_name=args.district_name,
                lat=args.lat,
                lon=args.lon,
                start_year=args.start_year,
                end_year=args.end_year,
                force_refresh=args.refresh
            )
        elif args.lat is not None and args.lon is not None:
            extract_point_lst_series(
                lat=args.lat,
                lon=args.lon,
                start_year=args.start_year,
                end_year=args.end_year,
                force_refresh=args.refresh
            )
        else:
            print("Please provide either --district_id and --district_name, or --lat and --lon.")
    except Exception as err:
        print(f"Extraction Error: {err}")
        sys.exit(1)
