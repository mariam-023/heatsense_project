"""
HeatSense Routes Module.
Handles all Flask routes including auth, location selection, GEE data,
analysis endpoints, mitigation, health risk, and reports.
"""

import os
import json
import requests
from flask import Blueprint, render_template, jsonify, request, session, redirect, url_for, Response

from app.database import get_db

main_bp = Blueprint('main', __name__)

import re
from werkzeug.security import generate_password_hash, check_password_hash

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _require_session():
    """Check if user is authenticated in the session."""
    return session.get('user') is not None

def _is_valid_email(email):
    """Validate email format for any valid domain."""
    pattern = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
    return bool(re.match(pattern, email.strip()))

def _is_valid_phone(phone):
    """Validate phone number format (allows digits, spaces, hyphens, optional + prefix)."""
    cleaned = re.sub(r'[\s\-()]', '', phone.strip())
    return bool(re.match(r'^\+?[0-9]{8,15}$', cleaned))

def _get_features_for_request():
    """
    Extract lat/lon from request args and return features + metadata.
    Supports both lat/lon params (new) and district_id (legacy).
    """
    from app.preprocessing import get_location_features, _lat_lon_fallback, _EE_INITIALIZED

    lat = request.args.get('lat', type=float)
    lon = request.args.get('lon', type=float)
    radius = request.args.get('radius', 15, type=float)

    if lat is not None and lon is not None:
        features, source = get_location_features(lat, lon, radius_km=radius)
        return features, lat, lon, source

    # Legacy: district_id fallback
    district_id = request.args.get('district_id', type=int)
    if district_id:
        db = get_db()
        district = db.execute("SELECT * FROM districts WHERE id = ?", (district_id,)).fetchone()
        if district:
            lat = district['latitude']
            lon = district['longitude']
            features, source = get_location_features(lat, lon, radius_km=15)
            return features, lat, lon, source

    return None, None, None, None


def _get_location_name():
    """Get location name from request params."""
    name = request.args.get('name', '')
    if name:
        return name
    district_id = request.args.get('district_id', type=int)
    if district_id:
        db = get_db()
        d = db.execute("SELECT name FROM districts WHERE id = ?", (district_id,)).fetchone()
        if d:
            return d['name']
    return 'Selected Location'


# ──────────────────────────────────────────────────────────────────────────────
# Page Routes
# ──────────────────────────────────────────────────────────────────────────────

@main_bp.route('/')
def root():
    """Entry point — redirect to splash screen."""
    return redirect(url_for('main.splash'))

@main_bp.route('/splash')
def splash():
    """Animated splash screen."""
    return render_template('splash.html')

@main_bp.route('/login')
def login():
    """Login and registration page."""
    if _require_session():
        return redirect(url_for('main.location_page'))
    return render_template('login.html', active_tab=request.args.get('tab', 'login'))

@main_bp.route('/register')
def register():
    """Direct route for registration view."""
    if _require_session():
        return redirect(url_for('main.location_page'))
    return render_template('login.html', active_tab='register')

@main_bp.route('/location')
def location_page():
    """Full-page interactive location selection screen."""
    if not _require_session():
        return redirect(url_for('main.login'))
    return render_template('location.html')

@main_bp.route('/dashboard')
def dashboard():
    """Main GIS analysis dashboard."""
    if not _require_session():
        return redirect(url_for('main.login'))

    # Pass location params from URL query string to template
    lat  = request.args.get('lat', 12.9716, type=float)
    lon  = request.args.get('lon', 77.5946, type=float)
    name = request.args.get('name', 'Bengaluru Urban')
    radius = request.args.get('radius', 15, type=float)
    location_id = request.args.get('location_id', type=int)

    db = get_db()
    districts = db.execute("SELECT * FROM districts").fetchall()
    user = session.get('user') or {'display_name': 'Guest', 'email': '', 'photo_url': ''}

    return render_template(
        'index.html',
        districts=districts,
        location_lat=lat,
        location_lon=lon,
        location_name=name,
        location_radius=radius,
        user=user,
    )

@main_bp.route('/alerts')
def alerts():
    """Health Risk & Alerts dashboard."""
    from app.health_risk import get_all_alerts, get_district_hhri, RISK_TIERS

    db = get_db()
    alerts_list = get_all_alerts(db)

    districts = db.execute('SELECT * FROM districts').fetchall()
    district_risk_cards = []
    for d in districts:
        try:
            hhri_result = get_district_hhri(dict(d))
            district_risk_cards.append({
                'name':         d['name'],
                'hhri':         hhri_result['hhri'],
                'risk_level':   hhri_result['risk_level'],
                'risk_icon':    hhri_result['risk_icon'],
                'colour_class': hhri_result['colour_class'],
                'badge_bg':     hhri_result['badge_bg'],
                'action':       hhri_result['action'],
                'advisory':     hhri_result['advisory'],
            })
        except Exception as e:
            print(f"[Alerts] HHRI calc failed for {d['name']}: {e}")

    tier_counts = {'Low': 0, 'Moderate': 0, 'High': 0, 'Very High': 0}
    for card in district_risk_cards:
        tier_counts[card['risk_level']] = tier_counts.get(card['risk_level'], 0) + 1

    return render_template(
        'alerts.html',
        alerts=alerts_list,
        district_risk_cards=district_risk_cards,
        tier_counts=tier_counts,
        risk_tiers=RISK_TIERS,
        districts=[dict(d) for d in districts],
    )

@main_bp.route('/logout')
def logout():
    """Clear session and redirect to splash."""
    session.clear()
    return redirect(url_for('main.splash'))


# ──────────────────────────────────────────────────────────────────────────────
# Auth API
# ──────────────────────────────────────────────────────────────────────────────

@main_bp.route('/api/auth/register', methods=['POST'])
def api_auth_register():
    """
    Registers a new user into the SQLite database.
    Validates name, email (any domain), phone, password, and confirmation.
    Stores securely hashed passwords.
    """
    body = request.get_json() or {}
    full_name = (body.get('full_name') or '').strip()
    email = (body.get('email') or '').strip().lower()
    phone = (body.get('phone_number') or '').strip()
    password = body.get('password') or ''
    confirm_password = body.get('confirm_password') or ''

    # 1. Validation: Required fields
    if not full_name:
        return jsonify({'success': False, 'message': 'Please enter your full name.'}), 400
    if len(full_name) < 2:
        return jsonify({'success': False, 'message': 'Full name must be at least 2 characters.'}), 400

    if not email:
        return jsonify({'success': False, 'message': 'Please enter your email address.'}), 400
    if not _is_valid_email(email):
        return jsonify({'success': False, 'message': 'Please enter a valid email address (e.g. name@domain.com).'}), 400

    if not phone:
        return jsonify({'success': False, 'message': 'Please enter your phone number.'}), 400
    if not _is_valid_phone(phone):
        return jsonify({'success': False, 'message': 'Please enter a valid phone number (10 to 15 digits).'}), 400

    if not password:
        return jsonify({'success': False, 'message': 'Please enter a password.'}), 400
    if len(password) < 6:
        return jsonify({'success': False, 'message': 'Password must be at least 6 characters long.'}), 400

    if password != confirm_password:
        return jsonify({'success': False, 'message': 'Passwords do not match. Please retype and confirm.'}), 400

    db = get_db()

    # 2. Check if email already exists
    existing = db.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
    if existing:
        return jsonify({'success': False, 'message': 'An account with this email address already exists. Please sign in.'}), 409

    # 3. Hash password securely & insert user
    password_hash = generate_password_hash(password, method='pbkdf2:sha256')
    try:
        db.execute(
            "INSERT INTO users (full_name, email, phone_number, password_hash) VALUES (?, ?, ?, ?)",
            (full_name, email, phone, password_hash)
        )
        db.commit()
        return jsonify({
            'success': True,
            'message': 'Registration successful! You can now sign in with your email and password.'
        }), 201
    except Exception as e:
        db.rollback()
        return jsonify({'success': False, 'message': f'Registration failed: {str(e)}'}), 500


@main_bp.route('/api/auth/login', methods=['POST'])
def api_auth_login():
    """
    Authenticates a user against the SQLite database using email and password.
    Creates a secure session on success.
    """
    body = request.get_json() or {}
    email = (body.get('email') or '').strip().lower()
    password = body.get('password') or ''

    if not email or not password:
        return jsonify({'success': False, 'message': 'Please provide both email address and password.'}), 400

    if not _is_valid_email(email):
        return jsonify({'success': False, 'message': 'Please enter a valid email address.'}), 400

    db = get_db()
    user = db.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()

    if not user or not check_password_hash(user['password_hash'], password):
        return jsonify({'success': False, 'message': 'Invalid email address or password. Please try again.'}), 401

    # Establish session
    session.clear()
    session['user'] = {
        'id':           user['id'],
        'display_name': user['full_name'],
        'email':        user['email'],
        'phone_number': user['phone_number'],
    }
    session.permanent = True

    return jsonify({
        'success': True,
        'message': f'Welcome back, {user["full_name"]}!',
        'redirect': '/location'
    }), 200


# ──────────────────────────────────────────────────────────────────────────────
# Location & Geocoding API
# ──────────────────────────────────────────────────────────────────────────────

@main_bp.route('/api/geocode')
def api_geocode():
    """
    Reverse geocodes a lat/lon to an exact place name using Nominatim (OpenStreetMap).
    Uses building/street level zoom (18) to return specific landmarks, streets, and localities.
    """
    lat  = request.args.get('lat', type=float)
    lon  = request.args.get('lon', type=float)
    full = request.args.get('full', '0')

    if lat is None or lon is None:
        return jsonify({'error': 'Missing lat/lon'}), 400

    try:
        resp = requests.get(
            'https://nominatim.openstreetmap.org/reverse',
            params={
                'lat': lat, 'lon': lon,
                'format': 'json',
                'zoom': 18,  # Exact street and building level precision
                'addressdetails': 1,
            },
            headers={'User-Agent': 'HeatSense-Karnataka/2.0 (academic project)'},
            timeout=8
        )
        data = resp.json()
        addr = data.get('address', {})

        # Extract specific address elements
        amenity   = addr.get('amenity') or addr.get('building') or addr.get('shop') or addr.get('office') or addr.get('tourism') or addr.get('leisure')
        road      = addr.get('road') or addr.get('pedestrian') or addr.get('street') or addr.get('residential')
        locality  = addr.get('neighbourhood') or addr.get('suburb') or addr.get('hamlet') or addr.get('village') or addr.get('town')
        city      = addr.get('city') or addr.get('city_district') or addr.get('county') or addr.get('state_district')
        raw_title = data.get('name')

        # Determine the most accurate primary name
        primary = raw_title or amenity or road or locality or city or f"{lat:.5f}, {lon:.5f}"

        # Build clean contextual name (e.g. "Urva Market, Mangaluru" or "Kuloor Ferry Road, Urva, Mangaluru")
        parts = [primary]
        if locality and locality != primary and locality != city:
            parts.append(locality)
        if city and city != primary and city not in parts:
            parts.append(city)

        display_short = ", ".join(parts) if parts else (data.get('display_name') or f"{lat:.5f}, {lon:.5f}")

        district = (
            addr.get('county') or
            addr.get('city') or
            addr.get('state_district') or '—'
        )
        state   = addr.get('state', 'Karnataka')
        p_type  = data.get('type', 'place').capitalize()
        level   = data.get('addresstype', 'Location').capitalize()

        if full == '1':
            return jsonify({
                'name':         display_short,
                'primary':      primary,
                'district':     district,
                'state':        state,
                'road':         road or '',
                'locality':     locality or '',
                'city':         city or '',
                'postcode':     addr.get('postcode', ''),
                'type':         p_type,
                'level':        level,
                'display_name': data.get('display_name', display_short),
                'lat':          lat,
                'lon':          lon,
            })

        return jsonify({'display_name': display_short, 'name': display_short})

    except Exception as e:
        print(f"[Geocode] Nominatim failed: {e}")
        return jsonify({'display_name': f"{lat:.5f}, {lon:.5f}", 'name': f"{lat:.5f}, {lon:.5f}"})


@main_bp.route('/api/detect-ip-location')
def api_detect_ip_location():
    """
    Fallback location detection via IP lookup when hardware GPS is unavailable or blocked.
    """
    try:
        resp = requests.get('http://ip-api.com/json/', timeout=5)
        data = resp.json()
        if data.get('status') == 'success':
            lat = float(data.get('lat', 12.9716))
            lon = float(data.get('lon', 77.5946))
            city = data.get('city', 'Karnataka')
            region = data.get('regionName', 'Karnataka')
            return jsonify({
                'success': True,
                'lat': lat,
                'lon': lon,
                'name': f"{city}, {region}",
                'accuracy': 5000,
                'source': 'ip'
            })
    except Exception as e:
        print(f"[IP Location] Detection failed: {e}")

    return jsonify({
        'success': False,
        'lat': 12.9716,
        'lon': 77.5946,
        'name': 'Bengaluru, Karnataka',
        'source': 'default'
    })


@main_bp.route('/api/search-location')
def api_search_location():
    """
    Searches for locations in Karnataka by text using Nominatim.
    Prioritizes specific place names/landmarks and returns up to 10 matching results.
    """
    q = request.args.get('q', '').strip()
    if len(q) < 2:
        return jsonify([])

    try:
        resp = requests.get(
            'https://nominatim.openstreetmap.org/search',
            params={
                'q':              f"{q}, Karnataka, India",
                'format':         'json',
                'addressdetails': 1,
                'limit':          10,
                'countrycodes':   'in',
            },
            headers={'User-Agent': 'HeatSense-Karnataka/2.0 (academic project)'},
            timeout=8
        )
        results = resp.json()
        out = []
        for r in results:
            addr = r.get('address', {})
            lat_f = float(r.get('lat', 0))
            lon_f = float(r.get('lon', 0))

            # Filter to Karnataka: check state name OR coordinates in Karnataka bounding box (11.5–18.6°N, 74.0–78.6°E)
            is_karnataka = (
                'Karnataka' in r.get('display_name', '') or
                addr.get('state') == 'Karnataka' or
                (11.5 <= lat_f <= 18.6 and 74.0 <= lon_f <= 78.6)
            )
            if not is_karnataka:
                continue

            raw_name = r.get('name')
            amenity  = addr.get('amenity') or addr.get('building') or addr.get('shop') or addr.get('office')
            road     = addr.get('road') or addr.get('street')
            locality = addr.get('suburb') or addr.get('neighbourhood') or addr.get('village') or addr.get('town')
            city     = addr.get('city') or addr.get('county') or addr.get('state_district')

            primary = raw_name or amenity or road or locality or city or r.get('display_name', '')
            parts = [primary]
            if locality and locality != primary and locality != city:
                parts.append(locality)
            if city and city != primary and city not in parts:
                parts.append(city)

            name = ", ".join(parts)

            out.append({
                'name':         name,
                'display_name': r.get('display_name', ''),
                'lat':          float(r['lat']),
                'lon':          float(r['lon']),
                'type':         r.get('type', 'place'),
            })
        return jsonify(out)

    except Exception as e:
        print(f"[Search] Nominatim search failed: {e}")
        return jsonify([])


@main_bp.route('/api/analyze-location', methods=['POST'])
def api_analyze_location():
    """
    Accepts a selected lat/lon, stores it (or finds existing),
    and returns a location_id for dashboard use.
    """
    body = request.get_json() or {}
    lat  = body.get('lat')
    lon  = body.get('lon')
    name = body.get('name', 'Unknown Location')
    radius_km = body.get('radius_km', 15)

    if lat is None or lon is None:
        return jsonify({'error': 'Missing lat/lon'}), 400

    db = get_db()

    # Try to find existing location within ~5km
    existing = db.execute(
        "SELECT id FROM locations WHERE ABS(latitude - ?) < 0.05 AND ABS(longitude - ?) < 0.05",
        (lat, lon)
    ).fetchone()

    if existing:
        return jsonify({'location_id': existing['id'], 'name': name})

    # Insert new location
    cursor = db.execute(
        "INSERT INTO locations (name, latitude, longitude, radius_km) VALUES (?, ?, ?, ?)",
        (name, lat, lon, radius_km)
    )
    db.commit()
    return jsonify({'location_id': cursor.lastrowid, 'name': name})


# ──────────────────────────────────────────────────────────────────────────────
# GEE Status
# ──────────────────────────────────────────────────────────────────────────────

@main_bp.route('/api/gee-status')
def api_gee_status():
    """
    Returns actual GEE connection status. Never hardcodes 'Active' if disconnected.
    """
    from app.preprocessing import get_gee_status
    return jsonify(get_gee_status())


# ──────────────────────────────────────────────────────────────────────────────
# Map Layers
# ──────────────────────────────────────────────────────────────────────────────

@main_bp.route('/api/map-layers')
def api_map_layers():
    """
    Renders and serves the interactive map HTML.
    Accepts lat/lon for any location, falls back to Karnataka center.
    """
    start_date = request.args.get('start_date', '2024-03-01')
    end_date   = request.args.get('end_date', '2024-05-31')
    lat        = request.args.get('lat', 14.5, type=float)
    lon        = request.args.get('lon', 75.7, type=float)
    radius_km  = request.args.get('radius', 15, type=float)
    mode       = request.args.get('mode', 'full')

    from app.visualization import generate_geemap_html
    html_content = generate_geemap_html(start_date, end_date, lat, lon, radius_km, mode=mode)
    return html_content


# ──────────────────────────────────────────────────────────────────────────────
# Analysis APIs
# ──────────────────────────────────────────────────────────────────────────────

@main_bp.route('/api/predict', methods=['GET', 'POST'])
def api_predict():
    """
    ML-powered endpoint — predicts CHI for any location (lat/lon or district_id).
    Returns current-day and future (+1.5°C scenario) predictions.
    """
    from app.ml import predict_current_conditions, predict_future_conditions, chi_to_risk_level
    from app.preprocessing import get_location_features

    features, lat, lon, source = _get_features_for_request()
    location_name = _get_location_name()

    if features is None:
        # Defaults to Bengaluru if nothing specified
        lat, lon = 12.9716, 77.5946
        from app.preprocessing import _lat_lon_fallback
        features = _lat_lon_fallback(lat, lon)
        source = 'climatological'
        location_name = 'Bengaluru Urban'

    current_chi  = predict_current_conditions(features)
    current_risk = chi_to_risk_level(current_chi)

    future_result = predict_future_conditions(features, temp_offset=1.5, ndvi_offset=-0.10, ndbi_offset=0.05)
    future_chi    = future_result['predicted_chi']
    future_risk   = chi_to_risk_level(future_chi)

    advisories = {
        'Low':       'Normal conditions. No immediate heat stress expected.',
        'Moderate':  'Monitor conditions. Vulnerable groups should stay hydrated.',
        'High':      'High heat stress. Limit outdoor activity during peak hours.',
        'Very High': 'Extreme heat event. Avoid outdoor exposure. Follow district alerts.'
    }

    # ── Integrated Overall Temperature ────────────────────────────────────
    # The "overall temperature" for the area is the ERA5-Land 2m air temperature,
    # which is the standard meteorological ambient temperature — the same value
    # that Google Weather, weather stations, and forecasts report.
    # LST (Land Surface Temp) is kept separate as it measures ground/rooftop
    # temperature from satellite, which is typically 5–15°C higher.
    air_temp_val = features.get('air_temp', 30.0)
    humidity_val = features.get('relative_humidity', 50.0)

    # Overall temperature = ERA5-Land 2m air temperature (matches Google Weather)
    overall_temp = round(air_temp_val, 1)

    # Apparent / "Feels Like" temperature
    if features.get('feels_like') is not None:
        feels_like = round(features['feels_like'], 1)
    else:
        # Heat Index using Rothfusz regression (NWS standard)
        T = air_temp_val * 9.0 / 5.0 + 32.0  # Convert to Fahrenheit for formula
        RH = humidity_val
        if T >= 80.0 and RH >= 40.0:
            HI = (-42.379 + 2.04901523 * T + 10.14333127 * RH
                   - 0.22475541 * T * RH - 0.00683783 * T * T
                   - 0.05481717 * RH * RH + 0.00122874 * T * T * RH
                   + 0.00085282 * T * RH * RH - 0.00000199 * T * T * RH * RH)
            feels_like = round((HI - 32.0) * 5.0 / 9.0, 1)  # Back to Celsius
        else:
            feels_like = overall_temp

    return jsonify({
        'location':             location_name,
        'lat':                  lat,
        'lon':                  lon,
        'data_source':          source,
        # Current conditions
        'current_chi':          current_chi,
        'current_risk_level':   current_risk,
        'current_advisory':     advisories[current_risk],
        'predicted_lst_celsius': round(features.get('lst', 32.0), 1),
        'overall_temperature_celsius': overall_temp,
        'feels_like_celsius':   feels_like,
        # Environmental factors
        'features':             {k: round(v, 3) for k, v in features.items()},
        # Future scenario
        'future_chi':           future_chi,
        'future_risk_level':    future_risk,
        'future_advisory':      advisories[future_risk],
        'scenario_label':       '+1.5°C Warming / -10% Green Cover',
        # Legacy fields
        'risk_level':           current_risk,
        'advisory':             advisories[current_risk],
        'uhi_index':            round(current_chi * 4, 2),
    })


# ──────────────────────────────────────────────────────────────────────────────
# Prophet Time-Series Forecasting API
# ──────────────────────────────────────────────────────────────────────────────

import threading

_PROPHET_LOCK = threading.Lock()
_CACHED_HISTORICAL_RECORDS = None
_CACHED_FORECAST_RECORDS = None
_CACHED_CSV_MTIME = 0

@main_bp.route('/api/forecast', methods=['GET'])
def get_prophet_forecast():
    """
    Get Prophet forecast and historical observations.
    Query params:
      - district_id: integer district ID from database
      - lat, lon: float latitude and longitude coordinates
      - name: location name
      - location: location slug or name (e.g. 'regional', 'bengaluru_urban', 'kalaburagi')
      - baseline: 'true' (force regional baseline) or 'false'
      - format: 'json' (default) or 'csv'
      - include_historical: 'true' (default) or 'false'
    """
    global _CACHED_HISTORICAL_RECORDS, _CACHED_FORECAST_RECORDS, _CACHED_CSV_MTIME
    try:
        from prophet_forecasting_module.backend.prophet_model import (
            load_and_preprocess_historical_data,
            run_prophet_forecast,
            get_historical_file_for_location,
            get_prediction_file_for_location,
            OUTPUT_PREDICTION_FILE
        )
        from prophet_forecasting_module.backend.gee_extractor import (
            extract_district_lst_series,
            extract_point_lst_series
        )
        import pandas as pd

        req_format = request.args.get("format", "json").lower()
        include_hist = request.args.get("include_historical", "true").lower() == "true"
        force_baseline = request.args.get("baseline", "false").lower() == "true"

        district_id = request.args.get("district_id", type=int)
        lat = request.args.get("lat", type=float)
        lon = request.args.get("lon", type=float)
        name = request.args.get("name", "").strip()
        req_location = request.args.get("location", "").strip()

        db = get_db()
        district = None

        # 1. Resolve district by explicit district_id or numeric location string
        if district_id:
            district = db.execute("SELECT * FROM districts WHERE id = ?", (district_id,)).fetchone()
        elif req_location and req_location.isdigit():
            district_id = int(req_location)
            district = db.execute("SELECT * FROM districts WHERE id = ?", (district_id,)).fetchone()

        # 2. Check if req_location or name or (lat, lon) match a district in DB
        if not district:
            if req_location and req_location.lower() not in ('regional', 'statewide', 'karnataka'):
                loc_clean = req_location.replace('_', ' ').strip()
                district = db.execute("SELECT * FROM districts WHERE LOWER(name) = LOWER(?)", (loc_clean,)).fetchone()
            if not district and name:
                district = db.execute("SELECT * FROM districts WHERE LOWER(name) = LOWER(?)", (name,)).fetchone()
            if not district and lat is not None and lon is not None:
                # Find nearest district center within ~15km (0.15 degrees)
                districts = db.execute("SELECT * FROM districts").fetchall()
                best_d = None
                best_dist = 0.15
                for d in districts:
                    dist_deg = ((d['latitude'] - lat)**2 + (d['longitude'] - lon)**2)**0.5
                    if dist_deg < best_dist:
                        best_dist = dist_deg
                        best_d = d
                if best_d:
                    district = best_d

        loc_slug = (req_location or name or '').lower().replace(' ', '_').replace('-', '_')
        is_regional = force_baseline or loc_slug in ('regional', 'statewide', 'karnataka')

        with _PROPHET_LOCK:
            target_key = "regional"
            spatial_unit = "Statewide Regional Baseline"
            location_label = "Karnataka Statewide (Regional Baseline)"
            data_scope = "regional_statewide"
            notice = ""
            has_local_data = False

            if not is_regional:
                if district:
                    district_id = district['id']
                    target_key = f"district_{district_id}"
                    hist_file = get_historical_file_for_location(target_key)
                    if hist_file is None or not hist_file.exists():
                        try:
                            print(f"[Prophet API] Triggering GEE historical extraction for District {district['id']} ({district['name']})...")
                            extract_district_lst_series(
                                district_id=district['id'],
                                district_name=district['name'],
                                lat=district['latitude'],
                                lon=district['longitude']
                            )
                            hist_file = get_historical_file_for_location(target_key)
                        except Exception as gee_err:
                            print(f"[Prophet API Warning] GEE district extraction failed for {district['name']}: {gee_err}")

                    if hist_file and hist_file.exists():
                        has_local_data = True
                        data_scope = "district_extracted"
                        spatial_unit = "District GAUL Polygon"
                        location_label = district['name']
                    else:
                        target_key = "regional"
                        data_scope = "regional_fallback"
                        spatial_unit = "Statewide Regional Baseline"
                        location_label = "Karnataka Statewide (Regional Baseline)"
                        notice = f"Satellite time-series for district '{district['name']}' unavailable. Displaying Karnataka Statewide (Regional Baseline)."

                elif lat is not None and lon is not None:
                    lat_key = f"{lat:.2f}"
                    lon_key = f"{lon:.2f}"
                    target_key = f"point_{lat_key}_{lon_key}"
                    hist_file = get_historical_file_for_location(target_key)
                    if hist_file is None or not hist_file.exists():
                        try:
                            print(f"[Prophet API] Triggering GEE historical point extraction for ({lat:.4f}, {lon:.4f})...")
                            extract_point_lst_series(
                                lat=lat,
                                lon=lon,
                                location_name=name or f"{lat:.2f}, {lon:.2f}"
                            )
                            hist_file = get_historical_file_for_location(target_key)
                        except Exception as gee_err:
                            print(f"[Prophet API Warning] GEE point extraction failed for ({lat}, {lon}): {gee_err}")

                    if hist_file and hist_file.exists():
                        has_local_data = True
                        data_scope = "point_extracted"
                        spatial_unit = "1km MODIS Satellite Pixel Buffer"
                        location_label = f"{name} (1km MODIS Pixel)" if name else f"Point ({lat:.2f}, {lon:.2f})"
                    else:
                        target_key = "regional"
                        data_scope = "regional_fallback"
                        spatial_unit = "Statewide Regional Baseline"
                        location_label = "Karnataka Statewide (Regional Baseline)"
                        notice = f"Satellite time-series for point ({lat:.4f}, {lon:.4f}) unavailable. Displaying Karnataka Statewide (Regional Baseline)."

                elif loc_slug:
                    target_key = loc_slug
                    hist_file = get_historical_file_for_location(target_key)
                    if hist_file and hist_file.exists():
                        has_local_data = True
                        data_scope = "local_extracted"
                        spatial_unit = "Extracted Local Area"
                        location_label = req_location.replace('_', ' ').title()
                    else:
                        target_key = "regional"
                        data_scope = "regional_fallback"
                        spatial_unit = "Statewide Regional Baseline"
                        location_label = "Karnataka Statewide (Regional Baseline)"
                        notice = f"Local dataset for '{req_location}' unavailable. Displaying Karnataka Statewide (Regional Baseline)."

            target_hist_file = get_historical_file_for_location(target_key)
            target_pred_file = get_prediction_file_for_location(target_key)

            # Run prophet if output prediction file doesn't exist for the target historical dataset
            if target_pred_file is None or not target_pred_file.exists():
                print(f"[Prophet] Prediction file missing for '{target_key}'. Fitting Prophet model...")
                df_hist_target = load_and_preprocess_historical_data(location_key=target_key)
                run_prophet_forecast(df_hist_target, location_key=target_key)

            # Read prediction CSV
            df_pred = pd.read_csv(target_pred_file)

            # If CSV format requested
            if req_format == "csv":
                csv_str = df_pred.to_csv(index=False)
                return Response(csv_str, mimetype="text/csv", headers={"Content-Disposition": f"attachment;filename=prediction_{target_key}.csv"})

            forecast_records = []
            for _, row in df_pred.iterrows():
                record = {
                    "ds": str(row["ds"]),
                    "yhat": round(float(row["yhat"]), 2)
                }
                if "yhat_lower" in row and pd.notnull(row["yhat_lower"]):
                    record["yhat_lower"] = round(float(row["yhat_lower"]), 2)
                if "yhat_upper" in row and pd.notnull(row["yhat_upper"]):
                    record["yhat_upper"] = round(float(row["yhat_upper"]), 2)
                forecast_records.append(record)

            historical_records = []
            if include_hist:
                try:
                    df_hist = load_and_preprocess_historical_data(location_key=target_key)
                    historical_records = [
                        {"ds": row["ds"].strftime("%Y-%m-%d"), "temp": round(float(row["y"]), 2)}
                        for _, row in df_hist.iterrows()
                    ]
                except Exception as e:
                    print(f"[Prophet Warning] Failed to load historical series for {target_key}: {e}")

        return jsonify({
            "status": "success",
            "district_id": district_id,
            "lat": lat,
            "lon": lon,
            "requested_location": req_location or name or "Statewide",
            "location_label": location_label,
            "spatial_unit": spatial_unit,
            "data_scope": data_scope,
            "has_local_data": has_local_data if not is_regional else True,
            "notice": notice,
            "historical_count": len(historical_records),
            "forecast_count": len(forecast_records),
            "historical": historical_records,
            "forecast": forecast_records
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[Prophet API Error] /api/forecast failed: {e}")
        return jsonify({"status": "error", "message": f"Forecast service error: {str(e)}"}), 500


@main_bp.route('/api/forecast/generate', methods=['POST'])
def generate_prophet_forecast_endpoint():
    """Re-run the Prophet model training & prediction pipeline on demand for a location."""
    global _CACHED_HISTORICAL_RECORDS, _CACHED_FORECAST_RECORDS, _CACHED_CSV_MTIME
    try:
        from prophet_forecasting_module.backend.prophet_model import (
            load_and_preprocess_historical_data,
            run_prophet_forecast,
            get_historical_file_for_location
        )
        req_location = request.args.get("location", "regional").strip()
        loc_slug = req_location.lower().replace(' ', '_').replace('-', '_')

        hist_file = get_historical_file_for_location(loc_slug)
        if loc_slug not in ('regional', 'statewide', 'karnataka', '') and (hist_file is None or not hist_file.exists()):
            return jsonify({
                "status": "error",
                "message": f"Cannot generate local forecast for '{req_location}': No authentic GEE historical observation dataset found. Run GEE extraction first."
            }), 400

        with _PROPHET_LOCK:
            periods = int(request.args.get("periods", 3650))
            df_hist = load_and_preprocess_historical_data(location_key=loc_slug)
            forecast_future, _ = run_prophet_forecast(df_hist, periods=periods, location_key=loc_slug)
            _CACHED_FORECAST_RECORDS = None
            _CACHED_CSV_MTIME = 0

        return jsonify({
            "status": "success",
            "location": loc_slug,
            "message": f"Successfully generated {len(forecast_future)} prediction steps for '{req_location}' with Meta Prophet.",
            "forecast_count": len(forecast_future),
            "start_date": str(forecast_future["ds"].min()),
            "end_date": str(forecast_future["ds"].max())
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[Prophet API Error] /api/forecast/generate failed: {e}")
        return jsonify({"status": "error", "message": f"Forecast generation failed: {str(e)}"}), 500



@main_bp.route('/api/forecast/health', methods=['GET'])
@main_bp.route('/api/health', methods=['GET'])
def prophet_health_check():
    """Prophet forecasting system health check endpoint."""
    try:
        from prophet_forecasting_module.backend.prophet_model import (
            INPUT_TS_FILE,
            FALLBACK_TS_FILE,
            OUTPUT_PREDICTION_FILE
        )
        input_file = INPUT_TS_FILE if INPUT_TS_FILE.exists() else FALLBACK_TS_FILE
        return jsonify({
            "status": "healthy",
            "service": "Prophet Forecasting Engine",
            "input_dataset_exists": input_file.exists(),
            "input_dataset_path": str(input_file.name),
            "output_prediction_exists": OUTPUT_PREDICTION_FILE.exists(),
            "output_prediction_path": str(OUTPUT_PREDICTION_FILE.name)
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@main_bp.route('/api/prediction-timeline')
def api_prediction_timeline():
    """Adapter endpoint providing legacy timeline callers access to Prophet forecast data."""
    return get_prophet_forecast()


@main_bp.route('/api/history')
def api_history():
    """
    Retrieves historical heat trends for any location (lat/lon or district_id).
    """
    from app.preprocessing import get_location_historical_trend

    lat = request.args.get('lat', type=float)
    lon = request.args.get('lon', type=float)
    name = request.args.get('name', 'Selected Location')
    start_year = request.args.get('start_year', 2015, type=int)
    end_year   = request.args.get('end_year', 2025, type=int)

    if lat is None or lon is None:
        # Legacy district_id
        district_id = request.args.get('district_id', type=int)
        if district_id:
            db = get_db()
            d = db.execute("SELECT * FROM districts WHERE id = ?", (district_id,)).fetchone()
            if d:
                lat, lon, name = d['latitude'], d['longitude'], d['name']
        if lat is None:
            lat, lon = 12.9716, 77.5946
            name = 'Bengaluru Urban'

    historical_data = get_location_historical_trend(lat, lon, name, start_year, end_year)

    from app.visualization import generate_plotly_temperature_trends
    plotly_json = generate_plotly_temperature_trends(historical_data, location_name=name)

    # Also return raw data for client-side use
    result = json.loads(plotly_json)
    return jsonify({
        'chart': result,
        'data': historical_data,
        'location': name,
        'lat': lat, 'lon': lon,
    })


@main_bp.route('/api/factor-analysis')
def api_factor_analysis():
    """
    Factor Analysis endpoint — returns radar chart + bar/pie/table charts.
    Uses actual processed feature values for the selected location.
    """
    from app.ml import get_feature_analysis
    from app.visualization import generate_radar_chart

    features, lat, lon, source = _get_features_for_request()
    location_name = _get_location_name()

    if features is None:
        from app.preprocessing import _lat_lon_fallback
        lat, lon = 12.9716, 77.5946
        features = _lat_lon_fallback(lat, lon)
        location_name = 'Bengaluru Urban'
        source = 'climatological'

    # Radar chart from actual feature values
    radar_json = generate_radar_chart(features, location_name=location_name)

    # Bar/pie/table from RF model feature importances × actual location values
    try:
        ml_analysis = get_feature_analysis(location_features=features)
    except Exception as e:
        print(f"[FactorAnalysis] get_feature_analysis error: {e}")
        ml_analysis = {'ranked_factors': [], 'bar_chart_json': '{}', 'pie_chart_json': '{}', 'table_chart_json': '{}'}

    # Build factor scores for display
    factor_scores = _build_factor_scores(features)

    return jsonify({
        **ml_analysis,
        'radar_chart_json': radar_json,
        'factor_scores':    factor_scores,
        'features':         {k: round(v, 3) for k, v in features.items()},
        'data_source':      source,
        'location':         location_name,
        'lat':              lat,
        'lon':              lon,
    })


def _build_factor_scores(features: dict) -> list:
    """Build 0–100 normalized factor scores for display cards."""
    lst  = features.get('lst', 30)
    ndvi = features.get('ndvi', 0.4)
    ndbi = features.get('ndbi', 0.1)
    air  = features.get('air_temp', 28)
    rh   = features.get('relative_humidity', 60)
    wind = features.get('wind_speed', 3)
    lulc = features.get('lulc_heat', 0.5)

    return [
        {'key': 'lst',   'label': 'Land Surface Temp',   'unit': '°C',  'value': round(lst,1),  'score': round(max(0,min(100,(lst-20)/30*100)),1),  'direction': 'heat',    'icon': '🌡️',  'color': '#ef4444'},
        {'key': 'air',   'label': 'Air Temperature',      'unit': '°C',  'value': round(air,1),  'score': round(max(0,min(100,(air-15)/30*100)),1),  'direction': 'heat',    'icon': '🌬️', 'color': '#f97316'},
        {'key': 'ndbi',  'label': 'Built-Up (NDBI)',      'unit': '',    'value': round(ndbi,3), 'score': round(max(0,min(100,(ndbi+0.5)/1.0*100)),1), 'direction': 'heat',  'icon': '🏙️', 'color': '#c0392b'},
        {'key': 'lulc',  'label': 'LULC Heat Score',      'unit': '',    'value': round(lulc,3), 'score': round(max(0,min(100,lulc*100)),1),          'direction': 'heat',    'icon': '🗺️',  'color': '#e67e22'},
        {'key': 'rh',    'label': 'Humidity',              'unit': '%',   'value': round(rh,1),   'score': round(max(0,min(100,(rh-10)/90*100)),1),    'direction': 'heat',    'icon': '💧',  'color': '#2980b9'},
        {'key': 'ndvi',  'label': 'Vegetation (NDVI)',     'unit': '',    'value': round(ndvi,3), 'score': round(max(0,min(100,(1-(ndvi+0.1)/0.9)*100)),1), 'direction': 'cool', 'icon': '🌿', 'color': '#27ae60'},
        {'key': 'wind',  'label': 'Wind Speed',            'unit': 'm/s', 'value': round(wind,1), 'score': round(max(0,min(100,(1-wind/10)*100)),1),   'direction': 'cool',    'icon': '💨',  'color': '#8e44ad'},
    ]


@main_bp.route('/api/health-risk')
def api_health_risk():
    """
    Health Risk endpoint — returns HHRI, risk level, and age-specific precautions.
    Works for any lat/lon.
    """
    from app.health_risk import get_location_hhri, get_age_specific_precautions
    from app.ml import predict_current_conditions

    features, lat, lon, source = _get_features_for_request()
    location_name = _get_location_name()

    if features is None:
        from app.preprocessing import _lat_lon_fallback
        lat, lon = 12.9716, 77.5946
        features = _lat_lon_fallback(lat, lon)
        location_name = 'Bengaluru Urban'

    hhri_result = get_location_hhri(lat, lon, features)
    chi = predict_current_conditions(features)

    return jsonify({
        'location':         location_name,
        'lat':              lat,
        'lon':              lon,
        'data_source':      source,
        'chi':              chi,
        'hhri':             hhri_result['hhri'],
        'risk_level':       hhri_result['risk_level'],
        'risk_icon':        hhri_result['risk_icon'],
        'badge_bg':         hhri_result['badge_bg'],
        'advisory':         hhri_result['advisory'],
        'action':           hhri_result['action'],
        'components':       hhri_result['components'],
        'age_precautions':  hhri_result['age_precautions'],
        'emergency_contacts': [
            {'label': 'Arogya Sahayavani', 'number': '104'},
            {'label': 'Ambulance',          'number': '108'},
            {'label': 'Police',             'number': '100'},
            {'label': 'BBMP Heat Helpline', 'number': '080-22221188'},
        ],
    })


@main_bp.route('/api/mitigation', methods=['POST'])
def api_mitigation():
    """
    Mitigation Simulation endpoint.
    Accepts: lat, lon (or district_id), scenario_type
    Returns: full before/after simulation payload.
    """
    body        = request.get_json() or {}
    strategy    = body.get('scenario_type', 'vegetation_expansion')
    lat         = body.get('lat')
    lon         = body.get('lon')
    radius      = body.get('radius', 15)
    name        = body.get('name', 'Selected Location')
    district_id = body.get('district_id')

    from app.preprocessing import get_location_features, _lat_lon_fallback, _EE_INITIALIZED
    from app.mitigation import run_mitigation_simulation

    if lat is not None and lon is not None:
        features, source = get_location_features(lat, lon, radius_km=radius)
    elif district_id:
        db = get_db()
        district = db.execute('SELECT * FROM districts WHERE id = ?', (district_id,)).fetchone()
        if not district:
            return jsonify({'error': 'District not found'}), 404
        lat = district['latitude']
        lon = district['longitude']
        name = district['name']
        features, source = get_location_features(lat, lon, radius_km=15)
    else:
        return jsonify({'error': 'Missing lat/lon or district_id'}), 400

    try:
        result = run_mitigation_simulation(features, strategy)
        result['location'] = name
        result['lat'] = lat
        result['lon'] = lon
        result['data_source'] = source
        return jsonify(result)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'Simulation failed: {e}'}), 500


@main_bp.route('/api/before-after-maps')
def api_before_after_maps():
    """
    Generates before and after mitigation Folium maps.
    Returns JSON with before_html and after_html.
    """
    from app.mitigation import run_mitigation_simulation, apply_strategy_offsets
    from app.preprocessing import get_location_features, _lat_lon_fallback

    lat      = request.args.get('lat', type=float)
    lon      = request.args.get('lon', type=float)
    strategy = request.args.get('strategy', 'vegetation_expansion')
    radius   = request.args.get('radius', 15, type=float)

    if lat is None or lon is None:
        lat, lon = 12.9716, 77.5946

    features, source = get_location_features(lat, lon, radius_km=radius)
    after_features = apply_strategy_offsets(features, strategy)

    from app.visualization import generate_before_after_maps
    before_html, after_html = generate_before_after_maps(
        lat, lon, radius, strategy, features, after_features
    )

    # Return as JSON with HTML embedded
    return jsonify({
        'before_html': before_html,
        'after_html':  after_html,
        'data_source': source,
    })


@main_bp.route('/api/alerts')
def api_alerts_json():
    """JSON API endpoint to fetch seeded health alerts from the database."""
    from app.health_risk import get_all_alerts
    db = get_db()
    alerts_list = get_all_alerts(db)
    result = [{
        "id":               a["id"],
        "alert_date":       a["alert_date"],
        "risk_level":       a["risk_level"],
        "advisory_message": a["advisory_message"],
        "status":           a["status"],
        "district_name":    a["district_name"],
    } for a in alerts_list]
    return jsonify(result)


# ──────────────────────────────────────────────────────────────────────────────
# Reports
# ──────────────────────────────────────────────────────────────────────────────

@main_bp.route('/api/download-report')
def api_download_report():
    """Generates and returns a downloadable text report."""
    from datetime import datetime
    from app.ml import predict_current_conditions, predict_future_conditions, chi_to_risk_level
    from app.health_risk import get_location_hhri
    from app.preprocessing import get_location_historical_trend

    features, lat, lon, source = _get_features_for_request()
    location_name = _get_location_name()

    if features is None:
        from app.preprocessing import _lat_lon_fallback
        lat, lon = 12.9716, 77.5946
        features = _lat_lon_fallback(lat, lon)
        location_name = 'Bengaluru Urban'

    current_chi  = predict_current_conditions(features)
    current_risk = chi_to_risk_level(current_chi)
    hhri_result  = get_location_hhri(lat, lon, features)
    future_result = predict_future_conditions(features)
    future_chi   = future_result['predicted_chi']
    future_risk  = chi_to_risk_level(future_chi)
    trend        = get_location_historical_trend(lat, lon, location_name)
    trend_slope  = trend[-1]['lst_slope'] if trend else 0.18

    report = f"""# HEATSENSE UHI ANALYSIS & HEALTH ASSESSMENT REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Study Area: {location_name}, Karnataka, India
Coordinates: Lat {lat:.5f}, Lon {lon:.5f}
Data Source: {source}

=========================================

1. CURRENT ENVIRONMENTAL PROFILE
---------------------------------
- Land Surface Temperature (LST): {features['lst']:.1f} °C
- Air Temperature (2m): {features['air_temp']:.1f} °C
- Normalized Difference Vegetation (NDVI): {features['ndvi']:.3f}
- Built-up Index (NDBI): {features['ndbi']:.3f}
- Relative Humidity: {features['relative_humidity']:.1f}%
- Wind Speed: {features['wind_speed']:.1f} m/s
- Land Use/Cover Heat Rating: {features['lulc_heat']:.2f}

2. COMPOSITE HEAT INDEX (CHI) & PREDICTIONS
-------------------------------------------
- Current Composite Heat Index (CHI): {current_chi:.3f}
- Heat Risk Rating: {current_risk}
- Heat Health Risk Index (HHRI): {hhri_result['hhri']:.3f}

Forecasted Climate Scenario (+1.5°C warming, -10% Green Space):
- Projected Future CHI: {future_chi:.3f}
- Projected Risk Level: {future_risk}
- Primary Health Advisory: {hhri_result['advisory']}

3. HISTORICAL WARMING TREND
----------------------------
- Decadal Warming Slope: +{trend_slope:.3f}°C/year

4. MITIGATION RECOMMENDATIONS
-------------------------------
* Vegetation Expansion — Cooling potential: -1.8°C LST
* Green Roof Retrofit  — Cooling potential: -1.2°C LST
* Reduce Built-Up Area — Cooling potential: -2.0°C LST
* Urban Parks          — Cooling potential: -2.2°C LST

=========================================
EMERGENCY CONTACTS (Karnataka):
- Arogya Sahayavani: 104
- Ambulance: 108
- Police: 100
- BBMP Heat Helpline: 080-22221188
"""

    safe_name = location_name.replace(' ', '_').replace('/', '_')
    return Response(
        report,
        mimetype="text/markdown",
        headers={"Content-disposition": f"attachment; filename=HeatSense_{safe_name}_Report.md"}
    )


@main_bp.route('/api/download-pdf')
def api_download_pdf():
    """Renders a print-ready HTML page for PDF download."""
    from datetime import datetime
    from app.ml import predict_current_conditions, predict_future_conditions, chi_to_risk_level
    from app.health_risk import get_location_hhri

    features, lat, lon, source = _get_features_for_request()
    location_name = _get_location_name()

    if features is None:
        from app.preprocessing import _lat_lon_fallback
        lat, lon = 12.9716, 77.5946
        features = _lat_lon_fallback(lat, lon)
        location_name = 'Bengaluru Urban'

    current_chi  = predict_current_conditions(features)
    current_risk = chi_to_risk_level(current_chi)
    hhri_result  = get_location_hhri(lat, lon, features)
    future_result = predict_future_conditions(features)
    future_chi   = future_result['predicted_chi']
    future_risk  = chi_to_risk_level(future_chi)

    # Create a minimal district-like dict for template compatibility
    location_dict = {'name': location_name, 'latitude': lat, 'longitude': lon}

    return render_template(
        'pdf_report.html',
        district=location_dict,
        date_now=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        features=features,
        current_chi=current_chi,
        current_risk=current_risk,
        hhri=hhri_result,
        future_chi=future_chi,
        future_risk=future_risk,
    )


# ──────────────────────────────────────────────────────────────────────────────
# SMS Heat Alerts API (Firebase Admin & Firestore Integration)
# ──────────────────────────────────────────────────────────────────────────────

def _get_firestore_db():
    """
    Safely retrieves the Firestore client instance using Firebase Admin SDK.
    Returns (db_client, error_message).
    """
    try:
        import firebase_admin
        from firebase_admin import firestore
        if not firebase_admin._apps:
            return None, "Firebase Admin SDK is not initialized. Ensure GOOGLE_APPLICATION_CREDENTIALS or Firebase project is configured."
        db = firestore.client()
        return db, None
    except Exception as e:
        return None, f"Firestore connection unavailable: {str(e)}"

def _mask_phone_number(phone_str):
    """Formats phone number safely for UI display and logging (e.g. +91 ******5678)."""
    clean = re.sub(r'[^\d+]', '', phone_str or '')
    if len(clean) >= 10:
        prefix = clean[:3] if clean.startswith('+') else clean[:2]
        suffix = clean[-4:]
        masked_len = max(4, len(clean) - len(prefix) - len(suffix))
        return f"{prefix} {'*' * masked_len}{suffix}"
    return "*******"

@main_bp.route('/api/alerts/sms-status', methods=['GET'])
def api_alerts_sms_status():
    """
    Returns the current SMS Heat Alert subscription status for the logged-in user.
    Session authentication is strictly enforced.
    """
    user = session.get('user')
    if not user or not user.get('id'):
        return jsonify({
            'success': False,
            'authenticated': False,
            'subscribed': False,
            'message': 'Please sign in to view your SMS alert subscription.'
        }), 401

    user_id = str(user['id'])
    db, err = _get_firestore_db()
    if err or db is None:
        return jsonify({
            'success': False,
            'authenticated': True,
            'subscribed': False,
            'firestore_configured': False,
            'message': f"SMS alert service is currently unconfigured ({err})."
        }), 200

    try:
        doc_ref = db.collection('sms_subscriptions').document(user_id)
        doc = doc_ref.get()
        if not doc.exists:
            return jsonify({
                'success': True,
                'authenticated': True,
                'subscribed': False,
                'firestore_configured': True,
                'message': 'No active SMS heat alert subscription found.'
            }), 200

        data = doc.to_dict() or {}
        is_active = data.get('status') == 'active' and data.get('opt_in_consent') is True

        return jsonify({
            'success': True,
            'authenticated': True,
            'subscribed': is_active,
            'firestore_configured': True,
            'district_id': data.get('district_id'),
            'district_name': data.get('district_name'),
            'phone_masked': data.get('phone_masked', _mask_phone_number(data.get('phone_e164', ''))),
            'consent_statement': data.get('consent_statement'),
            'subscribed_at': str(data.get('subscribed_at', '')),
            'message': 'Subscription status loaded successfully.'
        }), 200

    except Exception as e:
        print(f"[SMS Status Error] {e}")
        return jsonify({
            'success': False,
            'authenticated': True,
            'subscribed': False,
            'firestore_configured': False,
            'message': f'Could not retrieve subscription status: {str(e)}'
        }), 500


@main_bp.route('/api/alerts/sms-subscribe', methods=['POST'])
def api_alerts_sms_subscribe():
    """
    Subscribes the logged-in user to SMS Heat Alerts.
    Enforces Flask session validation, input sanitization, district existence check,
    and explicit opt-in consent recording in Firestore.
    """
    user = session.get('user')
    if not user or not user.get('id'):
        return jsonify({
            'success': False,
            'message': 'Please sign in to subscribe to SMS heat alerts.'
        }), 401

    body = request.get_json() or {}
    phone_raw = (body.get('phone_number') or '').strip()
    district_id = body.get('district_id')
    opt_in_consent = body.get('opt_in_consent') is True

    # 1. Validate Consent Checkbox
    if not opt_in_consent:
        return jsonify({
            'success': False,
            'message': 'Explicit consent is required to opt in to SMS heat alerts.'
        }), 400

    # 2. Validate Phone Format
    clean_phone = re.sub(r'[^\d+]', '', phone_raw)
    if not clean_phone.startswith('+'):
        clean_phone = '+91' + clean_phone.lstrip('0')

    digits_only = re.sub(r'\D', '', clean_phone)
    if len(digits_only) < 10 or len(digits_only) > 15:
        return jsonify({
            'success': False,
            'message': 'Please enter a valid mobile number with country code (e.g. +91 98765 43210).'
        }), 400

    # 3. Validate District in SQLite database
    if not district_id:
        return jsonify({
            'success': False,
            'message': 'Please select a valid district for SMS alerts.'
        }), 400

    db_sqlite = get_db()
    district_row = db_sqlite.execute('SELECT id, name FROM districts WHERE id = ?', (district_id,)).fetchone()
    if not district_row:
        return jsonify({
            'success': False,
            'message': 'The selected district was not found in the Karnataka district database.'
        }), 400

    district_name = district_row['name']
    user_id = str(user['id'])
    masked_phone = _mask_phone_number(clean_phone)

    # 4. Check Firestore client connection
    firestore_db, err = _get_firestore_db()
    if err or firestore_db is None:
        print(f"[SMS Subscribe Error] Firestore unavailable: {err}")
        return jsonify({
            'success': False,
            'message': f'Subscription failed: Firestore service is unavailable on the server. ({err})'
        }), 503

    # 5. Persist evidence of consent in Firestore via Admin SDK
    try:
        from firebase_admin import firestore
        consent_statement = "I explicitly opt in to receive HeatSense SMS heat alerts for my selected district."
        doc_data = {
            'user_id': user['id'],
            'user_email': user.get('email', ''),
            'district_id': district_row['id'],
            'district_name': district_name,
            'phone_e164': clean_phone,
            'phone_masked': masked_phone,
            'opt_in_consent': True,
            'consent_statement': consent_statement,
            'consent_version': '1.0-2026',
            'status': 'active',
            'subscribed_at': firestore.SERVER_TIMESTAMP,
            'updated_at': firestore.SERVER_TIMESTAMP,
            'unsubscribed_at': None
        }

        firestore_db.collection('sms_subscriptions').document(user_id).set(doc_data, merge=True)
        print(f"[SMS Subscription] Saved active subscription for user {user_id} ({district_name}, {masked_phone})")

        return jsonify({
            'success': True,
            'subscribed': True,
            'district_name': district_name,
            'district_id': district_row['id'],
            'phone_masked': masked_phone,
            'message': f'Successfully subscribed to SMS Heat Alerts for {district_name} ({masked_phone})!'
        }), 200

    except Exception as e:
        print(f"[SMS Subscribe Exception] {e}")
        return jsonify({
            'success': False,
            'message': f'Failed to save subscription to database: {str(e)}'
        }), 500


@main_bp.route('/api/alerts/sms-unsubscribe', methods=['POST'])
def api_alerts_sms_unsubscribe():
    """
    Unsubscribes the logged-in user from SMS Heat Alerts.
    Updates status to 'unsubscribed' and sets opt_in_consent to False.
    """
    user = session.get('user')
    if not user or not user.get('id'):
        return jsonify({
            'success': False,
            'message': 'Please sign in to manage your subscription.'
        }), 401

    user_id = str(user['id'])
    firestore_db, err = _get_firestore_db()
    if err or firestore_db is None:
        return jsonify({
            'success': False,
            'message': f'Unsubscribe failed: Firestore service is unavailable ({err}).'
        }), 503

    try:
        from firebase_admin import firestore
        doc_ref = firestore_db.collection('sms_subscriptions').document(user_id)
        doc_ref.set({
            'status': 'unsubscribed',
            'opt_in_consent': False,
            'unsubscribed_at': firestore.SERVER_TIMESTAMP,
            'updated_at': firestore.SERVER_TIMESTAMP
        }, merge=True)

        print(f"[SMS Unsubscribe] User {user_id} unsubscribed successfully.")
        return jsonify({
            'success': True,
            'subscribed': False,
            'message': 'You have been unsubscribed from HeatSense SMS Heat Alerts.'
        }), 200

    except Exception as e:
        print(f"[SMS Unsubscribe Exception] {e}")
        return jsonify({
            'success': False,
            'message': f'Could not complete unsubscribe request: {str(e)}'
        }), 500
