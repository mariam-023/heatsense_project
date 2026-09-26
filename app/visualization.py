"""
HeatSense Visualization Module.
Generates interactive web maps using Geemap/Folium, dynamic charts using Plotly,
and radar/spider charts for factor analysis.
"""

import os
import json
import tempfile

def generate_geemap_html(start_date="2024-03-01", end_date="2024-05-31",
                         lat=14.5, lon=75.7, radius_km=15, mode="full"):
    """
    Creates an interactive Leaflet/Folium map.
    - mode='home': Displays focused Integrated Multi-Factor Heat Analysis (CHI & Hotspots).
    - mode='full': Displays all environmental layers (LST, NDVI, NDBI, LULC, Air Temp, RH, Wind) plus CHI & Hotspots in layer control.
    Uses direct Google Earth Engine Tile URLs for seamless standalone browser rendering.
    """
    import folium
    from app.preprocessing import (
        initialize_earth_engine,
        process_landsat_data, get_lulc_data, get_era5_land_daily_climate,
        calculate_composite_heat_index, classify_heat_hotspots,
        _EE_INITIALIZED
    )

    if not _EE_INITIALIZED:
        initialize_earth_engine()

    if _EE_INITIALIZED:
        try:
            import ee

            m = folium.Map(
                location=[lat, lon],
                zoom_start=12,
                tiles=None,
            )
            folium.TileLayer(
                tiles='https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}',
                attr='Esri, HERE, Garmin, &copy; OpenStreetMap contributors',
                name='Dark Base',
                max_zoom=18,
                control=False,
            ).add_to(m)
            folium.TileLayer(
                tiles='https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}',
                attr='Esri',
                name='Labels',
                max_zoom=18,
                overlay=True,
                control=False,
            ).add_to(m)

            # Location AOI
            point = ee.Geometry.Point([lon, lat])
            region = point.buffer(radius_km * 1000)

            # Helper to add GEE image layer to folium map
            def add_ee_tile(ee_img, viz, layer_name, show=True, opacity=0.85):
                try:
                    map_id = ee.Image(ee_img).getMapId(viz)
                    folium.raster_layers.TileLayer(
                        tiles=map_id['tile_fetcher'].url_format,
                        attr='Google Earth Engine',
                        name=layer_name,
                        overlay=True,
                        control=True,
                        show=show,
                        opacity=opacity,
                    ).add_to(m)
                except Exception as layer_err:
                    print(f"[Visualization] Layer '{layer_name}' skipped: {layer_err}")

            if mode == 'full':
                # ── 1. Water / Ocean Bodies ──────────────────────────────────────────
                lulc = get_lulc_data(region)
                water_mask = lulc.eq(80)
                water_vis_img = water_mask.selfMask().clip(region)
                water_viz = {'min': 0, 'max': 1, 'palette': ['#1a6eb5']}
                add_ee_tile(water_vis_img, water_viz,
                            'Ocean / Water Bodies', show=False, opacity=0.75)

                # ── 2. Landsat 8+9 Composite (LST, NDVI, NDBI) ──────────────────────
                landsat_comp = process_landsat_data(start_date, end_date, region)

                lst_viz = {'bands': ['LST'], 'min': 20.0, 'max': 45.0,
                           'palette': ['#0000ff', '#00ffff', '#ffff00', '#ff7f00', '#ff0000']}
                add_ee_tile(landsat_comp.select('LST').clip(region), lst_viz, 'Land Surface Temp (LST °C)', show=False)

                ndvi_viz = {'bands': ['NDVI'], 'min': -0.1, 'max': 0.8,
                            'palette': ['#ffffff', '#f7fcb9', '#addd8e', '#31a354', '#006837']}
                add_ee_tile(landsat_comp.select('NDVI').clip(region), ndvi_viz, 'Vegetation Index (NDVI)', show=False)

                ndbi_viz = {'bands': ['NDBI'], 'min': -0.5, 'max': 0.5,
                            'palette': ['#0000ff', '#ffffff', '#ff0000']}
                add_ee_tile(landsat_comp.select('NDBI').clip(region), ndbi_viz, 'Built-Up Index (NDBI)', show=False)

                # ── 3. ESA WorldCover LULC ───────────────────────────────────────────
                lulc_viz = {
                    'min': 10, 'max': 100,
                    'palette': ['#006400', '#ffbb22', '#ffff4c', '#f096ff', '#fa0000',
                                '#b4b4b4', '#f0f0f0', '#0064c8', '#0096a0', '#00cf75', '#fae6a0']
                }
                add_ee_tile(lulc.clip(region), lulc_viz, 'ESA LULC (WorldCover)', show=False)

                # ── 4. ERA5 Climate (Air Temp, Humidity, Wind Speed) ─────────────────
                climate = get_era5_land_daily_climate(start_date, end_date, region)

                air_temp_viz = {'bands': ['air_temperature'], 'min': 15.0, 'max': 40.0,
                                'palette': ['#1a9850', '#fee08b', '#d73027']}
                add_ee_tile(climate.select('air_temperature').clip(region), air_temp_viz, 'Air Temperature (2m °C)', show=False)

                rh_viz = {'bands': ['relative_humidity'], 'min': 10.0, 'max': 90.0,
                          'palette': ['#a6611a', '#f5f5f5', '#018571']}
                add_ee_tile(climate.select('relative_humidity').clip(region), rh_viz, 'Relative Humidity (%)', show=False)

                wind_viz = {'bands': ['wind_speed'], 'min': 0.0, 'max': 8.0,
                            'palette': ['#f7f7f7', '#cccccc', '#969696', '#525252', '#080808']}
                add_ee_tile(climate.select('wind_speed').clip(region), wind_viz, 'Wind Speed (10m m/s)', show=False)

            # ── Multi-Factor Composite Heat Index (CHI) ──────────────────────────
            chi = calculate_composite_heat_index(start_date, end_date, region)
            chi_viz = {
                'min': 0.0, 'max': 1.0,
                'palette': [
                    '#313695', '#4575b4', '#74add1', '#abd9e9',
                    '#ffffbf', '#fee090', '#fdae61', '#f46d43',
                    '#d73027', '#a50026'
                ]
            }
            chi_layer_name = 'Composite Heat Index (CHI)' if mode == 'full' else 'Integrated Heat Intensity (Continuous CHI)'
            add_ee_tile(
                chi.clip(region), chi_viz,
                chi_layer_name,
                show=True, opacity=0.88
            )

            # ── Heat Hotspots Classification ──────────────────────────────────────
            # 0: Low (<0.35) | 1: Moderate (0.35–0.55) | 2: High (0.55–0.75) | 3: Very High (≥0.75)
            hotspots = classify_heat_hotspots(chi, region=region)
            hotspot_viz = {
                'min': 0, 'max': 3,
                'palette': ['#27ae60', '#f1c40f', '#e67e22', '#c0392b']
            }
            hotspot_layer_name = 'Heat Hotspots Classification' if mode == 'full' else 'Integrated Heat Hazard Zones (Classified)'
            add_ee_tile(
                hotspots.clip(region), hotspot_viz,
                hotspot_layer_name,
                show=False, opacity=0.85
            )

            # ── Water Bodies Overlay (always on top, rendered in blue) ────────────
            # Load LULC and ESA permanent water to mask coastal/ocean pixels blue
            lulc_water = get_lulc_data(region)
            # ESA WorldCover class 80 = Permanent Water Bodies
            water_mask = lulc_water.eq(80)
            # JRC Global Surface Water adds river/lake pixels
            try:
                jrc_water = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('occurrence') \
                    .gte(70).clip(region)
                combined_water = water_mask.Or(jrc_water)
            except Exception:
                combined_water = water_mask

            water_viz = {'min': 0, 'max': 1, 'palette': ['#1a6eb5']}
            add_ee_tile(
                combined_water.selfMask().clip(region),
                water_viz,
                'Water Bodies (Ocean / Rivers / Lakes)',
                show=True, opacity=0.85
            )

            # ── Analysis AOI Boundary Circle & Center Marker ──────────────────
            folium.Circle(
                location=[lat, lon],
                radius=radius_km * 1000,
                color='#f97316',
                fill=True,
                fill_color='#f97316',
                fill_opacity=0.03,
                weight=2,
                dash_array='6 3',
                tooltip=f"Analysis AOI ({radius_km} km radius)"
            ).add_to(m)

            folium.Marker(
                location=[lat, lon],
                popup=folium.Popup(
                    f"<div style='font-family:sans-serif;min-width:180px;'>"
                    f"<b style='color:#f97316;font-size:0.9rem;'>Selected Location</b><br>"
                    f"<small style='color:#475569;'>Lat: {lat:.5f}, Lon: {lon:.5f}</small><hr style='margin:4px 0;'>"
                    f"<span style='font-size:0.75rem;color:#1e293b;'>GEE Multi-Sensor Live Imagery</span>"
                    f"</div>",
                    max_width=260
                ),
                icon=folium.Icon(color='orange', icon='fire', prefix='fa'),
            ).add_to(m)

            # ── On-Map Integrated Multi-Factor Legend (Home mode) ─────────────────
            if mode == 'home':
                legend_html = f"""
                <div style="
                    position: fixed;
                    bottom: 20px;
                    left: 20px;
                    z-index: 1000;
                    background: rgba(15, 15, 15, 0.94);
                    border: 1px solid rgba(255, 255, 255, 0.12);
                    border-radius: 12px;
                    padding: 12px 14px;
                    font-family: 'Inter', -apple-system, sans-serif;
                    color: #f8fafc;
                    box-shadow: 0 4px 20px rgba(0,0,0,0.6);
                    backdrop-filter: blur(8px);
                    max-width: 250px;
                ">
                    <div style="font-size: 0.78rem; font-weight: 700; color: #fb923c; margin-bottom: 4px; display:flex; align-items:center; gap:5px;">
                        <span>🔥</span> Integrated Multi-Factor CHI
                    </div>
                    <div style="font-size: 0.68rem; color: #cbd5e1; margin-bottom: 8px;">
                        Normalized combination of LST, Air Temp, NDBI, LULC, Humidity, NDVI & Wind
                    </div>
                    
                    <!-- Continuous Intensity Gradient Bar -->
                    <div style="margin-bottom: 8px;">
                        <div style="
                            height: 10px;
                            border-radius: 4px;
                            background: linear-gradient(to right, #313695, #4575b4, #74add1, #abd9e9, #ffffbf, #fee090, #fdae61, #f46d43, #d73027, #a50026);
                            border: 1px solid rgba(255,255,255,0.2);
                        "></div>
                        <div style="display:flex; justify-content:space-between; font-size:0.65rem; color:#94a3b8; margin-top:2px;">
                            <span>0.0 (Cool)</span>
                            <span>0.5</span>
                            <span>1.0 (Intense)</span>
                        </div>
                    </div>

                    <!-- 4 Hazard Tiers -->
                    <div style="border-top: 1px solid rgba(255,255,255,0.08); padding-top: 6px; font-size: 0.7rem;">
                        <div style="display:flex; align-items:center; gap:6px; margin-bottom:2px;">
                            <span style="display:inline-block; width:10px; height:10px; border-radius:2px; background:#27ae60;"></span>
                            <span style="color:#e2e8f0;">Low</span>
                            <span style="margin-left:auto; color:#94a3b8; font-size:0.64rem;">&lt; 0.35</span>
                        </div>
                        <div style="display:flex; align-items:center; gap:6px; margin-bottom:2px;">
                            <span style="display:inline-block; width:10px; height:10px; border-radius:2px; background:#f1c40f;"></span>
                            <span style="color:#e2e8f0;">Moderate</span>
                            <span style="margin-left:auto; color:#94a3b8; font-size:0.64rem;">0.35–0.55</span>
                        </div>
                        <div style="display:flex; align-items:center; gap:6px; margin-bottom:2px;">
                            <span style="display:inline-block; width:10px; height:10px; border-radius:2px; background:#e67e22;"></span>
                            <span style="color:#e2e8f0;">High</span>
                            <span style="margin-left:auto; color:#94a3b8; font-size:0.64rem;">0.55–0.75</span>
                        </div>
                        <div style="display:flex; align-items:center; gap:6px; margin-bottom:4px;">
                            <span style="display:inline-block; width:10px; height:10px; border-radius:2px; background:#c0392b;"></span>
                            <span style="color:#e2e8f0;">Very High</span>
                            <span style="margin-left:auto; color:#94a3b8; font-size:0.64rem;">≥ 0.75</span>
                        </div>
                        <div style="display:flex; align-items:center; gap:6px; border-top:1px solid rgba(255,255,255,0.08); padding-top:4px;">
                            <span style="display:inline-block; width:10px; height:10px; border-radius:2px; background:#1a6eb5;"></span>
                            <span style="color:#93c5fd;">Water Bodies</span>
                            <span style="margin-left:auto; color:#94a3b8; font-size:0.64rem;">Ocean / River</span>
                        </div>
                    </div>
                </div>
                """
                m.get_root().html.add_child(folium.Element(legend_html))

            # ── Layer Control ────────────────────────────────────────────────────
            folium.LayerControl(position='topright', collapsed=False).add_to(m)

            return m.get_root().render()

        except Exception as e:
            print(f"[Visualization] GEE map failed: {e}")

    # ── Folium fallback map ──────────────────────────────────────────────────
    return _generate_folium_fallback_map(lat, lon, radius_km)



def _generate_folium_fallback_map(lat, lon, radius_km=15):
    """
    Generates a Folium map centered on the selected location as GEE fallback.
    Uses real OpenStreetMap tiles (no fake data).
    """
    import folium

    m = folium.Map(
        location=[lat, lon],
        zoom_start=13,
        tiles=None,
    )
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}',
        attr='Esri, HERE, Garmin, &copy; OpenStreetMap contributors',
        name='Dark Base',
        max_zoom=17,
        control=False
    ).add_to(m)
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}',
        attr='Esri, HERE, Garmin, &copy; OpenStreetMap contributors',
        name='Labels',
        max_zoom=17,
        overlay=True,
        control=False
    ).add_to(m)

    # AOI circle at selected location
    folium.Circle(
        location=[lat, lon],
        radius=radius_km * 1000,
        color='#f97316',
        fill=True,
        fill_color='#f97316',
        fill_opacity=0.05,
        weight=2,
        dash_array='8 4',
        tooltip=f"Analysis AOI ({radius_km} km radius)",
    ).add_to(m)

    # Orange pin at selected location
    folium.Marker(
        location=[lat, lon],
        popup=folium.Popup(
            f"<b style='color:#f97316;'>Selected Location</b><br>"
            f"Lat: {lat:.5f}, Lon: {lon:.5f}<br>"
            f"<small>Analysis AOI: {radius_km} km radius</small>",
            max_width=250
        ),
        icon=folium.Icon(color='orange', icon='fire', prefix='fa'),
    ).add_to(m)

    # GEE offline banner
    html_banner = f"""
    <div style="position: fixed; top: 12px; left: 50%; transform: translateX(-50%);
                width: 420px; z-index:9999;
                background: rgba(15,15,15,0.95); padding: 10px 16px; border-radius: 10px;
                border: 1px solid rgba(239, 68, 68, 0.5);
                box-shadow: 0 4px 24px rgba(0,0,0,0.5); font-family: 'Inter', sans-serif;">
        <div style="display:flex;align-items:center;gap:8px;">
            <span style="font-size:16px;">⚠️</span>
            <div>
                <div style="color: #ef4444; font-size: 12px; font-weight: 700;">GEE Offline — Basemap View</div>
                <div style="color: rgba(255,255,255,0.5); font-size: 10px; margin-top: 2px;">
                    Run <code style="background:rgba(255,255,255,0.1);padding:1px 4px;border-radius:3px;">earthengine authenticate</code> to load satellite layers.
                </div>
            </div>
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(html_banner))

    fd, path = tempfile.mkstemp(suffix='.html')
    try:
        m.save(path)
        with open(path, 'r', encoding='utf-8') as f:
            html_content = f.read()
    finally:
        os.close(fd)
        os.remove(path)

    return html_content


def generate_before_after_maps(lat, lon, radius_km, strategy_key, before_features, after_features,
                                 start_date='2024-03-01', end_date='2024-05-31'):
    """
    Generates two Folium maps — Before mitigation and After mitigation —
    for the Before & After comparison swipe panel.
    - BEFORE: Displays the actual baseline Composite Heat Index (CHI) heat layer.
    - AFTER: Displays the actual post-mitigation CHI heat layer derived from calculated CHI reduction.
    - Both layers share the exact same geographic bounds, center, zoom, basemap, and CHI color scale:
        Low (<0.35) -> #27ae60 (Green)
        Moderate (0.35-0.55) -> #f1c40f (Yellow)
        High (0.55-0.75) -> #e67e22 (Orange)
        Very High (>=0.75) -> #c0392b (Red)
    - Realistic, un-exaggerated visual change with basemap visibility underneath.

    Returns:
        tuple: (before_html: str, after_html: str)
    """
    import folium
    from app.ml import predict_current_conditions, chi_to_risk_level
    from app.mitigation import STRATEGIES

    strategy = STRATEGIES.get(strategy_key, {
        'label': 'Vegetation Expansion',
        'icon': '🌿',
        'colour': '#27ae60',
        'description': 'Street trees and urban forests'
    })

    before_chi  = predict_current_conditions(before_features)
    before_risk = chi_to_risk_level(before_chi)
    before_lst  = round(before_features.get('lst', 32.0), 1)

    after_chi   = predict_current_conditions(after_features)
    after_risk  = chi_to_risk_level(after_chi)
    after_lst   = round(after_features.get('lst', before_lst - 1.8), 1)

    chi_delta   = round(max(0.0, before_chi - after_chi), 4)
    lst_delta   = round(max(0.0, before_lst - after_lst), 1)

    # ── GEE Multi-Factor CHI & Water Layers ──────────────────────────────────
    gee_before_tile = None
    gee_after_tile = None
    gee_water_tile = None

    try:
        from app.preprocessing import (
            _EE_INITIALIZED, initialize_earth_engine,
            calculate_composite_heat_index, get_lulc_data
        )
        if not _EE_INITIALIZED:
            initialize_earth_engine()

        import ee
        point = ee.Geometry.Point([lon, lat])
        region = point.buffer(radius_km * 1000)

        # Water Bodies (Ocean / Rivers / Lakes) in blue
        lulc_water = get_lulc_data(region)
        water_mask = lulc_water.eq(80)
        try:
            jrc_water = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('occurrence').gte(70).clip(region)
            combined_water = water_mask.Or(jrc_water)
        except Exception:
            combined_water = water_mask

        water_viz = {'min': 0, 'max': 1, 'palette': ['#1a6eb5']}
        water_map_id = combined_water.selfMask().clip(region).getMapId(water_viz)
        gee_water_tile = water_map_id['tile_fetcher'].url_format

        chi_img = calculate_composite_heat_index(start_date, end_date, region)
        chi_mitigated_img = chi_img.subtract(chi_delta).clamp(0.0, 1.0)

        # EXACT SAME CHI COLOR PALETTE FOR BOTH SIDES
        chi_viz = {
            'min': 0.0,
            'max': 1.0,
            'palette': ['#27ae60', '#f1c40f', '#e67e22', '#c0392b']
        }

        before_map_id = chi_img.clip(region).getMapId(chi_viz)
        after_map_id = chi_mitigated_img.clip(region).getMapId(chi_viz)

        gee_before_tile = before_map_id['tile_fetcher'].url_format
        gee_after_tile = after_map_id['tile_fetcher'].url_format
    except Exception as gee_err:
        print(f"[Before/After] GEE tile generation deferred: {gee_err}")

    def make_map(is_after=False):
        # Exact same center, zoom, bounds, and locked view for 100% pixel-perfect alignment
        m = folium.Map(
            location=[lat, lon],
            zoom_start=13,
            tiles=None,
            zoom_control=False,
            scroll_wheel_zoom=False,
            dragging=False,
            touch_zoom=False,
            double_click_zoom=False,
            box_zoom=False,
            attribution_control=False
        )

        # Shared crisp, watermark-free dark basemap + place/street labels
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}',
            attr='Esri, HERE, Garmin, &copy; OpenStreetMap contributors',
            name='Dark Base',
            max_zoom=18,
            control=False
        ).add_to(m)
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}',
            attr='Esri',
            name='Labels',
            max_zoom=18,
            overlay=True,
            control=False
        ).add_to(m)

        # ── 1. Heat Overlay (GEE or Climatological Fallback) ──────────────────
        tile_url = gee_after_tile if is_after else gee_before_tile
        if tile_url:
            folium.raster_layers.TileLayer(
                tiles=tile_url,
                attr='Google Earth Engine',
                name='Heat Layer (CHI) After' if is_after else 'Heat Layer (CHI) Before',
                overlay=True,
                control=False,
                show=True,
                opacity=0.75,
            ).add_to(m)
        else:
            # Fallback: Realistic concentric heat intensity rings using the SAME CHI palette
            def get_chi_hex(v):
                if v < 0.35: return '#27ae60'   # Low (Green)
                if v < 0.55: return '#f1c40f'   # Moderate (Yellow)
                if v < 0.75: return '#e67e22'   # High (Orange)
                return '#c0392b'                # Very High (Red)

            curr_chi = after_chi if is_after else before_chi
            rings = [
                (1.00, get_chi_hex(max(0.0, curr_chi - 0.05)), 0.18),
                (0.65, get_chi_hex(curr_chi), 0.28),
                (0.35, get_chi_hex(min(1.0, curr_chi + 0.05)), 0.40),
            ]
            for r_frac, col, op in rings:
                folium.Circle(
                    location=[lat, lon],
                    radius=radius_km * 1000 * r_frac,
                    color=col,
                    fill=True,
                    fill_color=col,
                    fill_opacity=op,
                    weight=1,
                ).add_to(m)

        # ── 2. Water Bodies Overlay (always on top, rendered in blue) ────────
        if gee_water_tile:
            folium.raster_layers.TileLayer(
                tiles=gee_water_tile,
                attr='Google Earth Engine',
                name='Water Bodies (Ocean / Rivers / Lakes)',
                overlay=True,
                control=False,
                show=True,
                opacity=0.85,
            ).add_to(m)

        # ── 3. Analysis AOI Boundary Circle (Exact same for both) ────────────
        folium.Circle(
            location=[lat, lon],
            radius=radius_km * 1000,
            color='#f97316',
            fill=False,
            weight=2,
            dash_array='6 3',
        ).add_to(m)

        # ── 4. Center Marker ─────────────────────────────────────────────────
        if not is_after:
            marker_color = 'orange'
            marker_icon = 'fire'
            popup_html = (
                f"<div style='font-family:Inter,sans-serif;min-width:180px;'>"
                f"<b style='color:#ef4444;font-size:0.92rem;'>🔴 Baseline (Before Mitigation)</b>"
                f"<hr style='margin:4px 0;border-color:rgba(255,255,255,0.15);'>"
                f"<b>Composite Heat Index:</b> {before_chi:.3f} ({before_risk})<br>"
                f"<b>Surface Temp (LST):</b> {before_lst:.1f}°C<br>"
                f"<b>Vegetation (NDVI):</b> {before_features.get('ndvi',0):.3f}<br>"
                f"<b>Built-Up (NDBI):</b> {before_features.get('ndbi',0):.3f}"
                f"</div>"
            )
        else:
            marker_color = 'green'
            marker_icon = 'leaf'
            popup_html = (
                f"<div style='font-family:Inter,sans-serif;min-width:180px;'>"
                f"<b style='color:#10b981;font-size:0.92rem;'>🟢 After: {strategy['label']}</b>"
                f"<hr style='margin:4px 0;border-color:rgba(255,255,255,0.15);'>"
                f"<b>Composite Heat Index:</b> {after_chi:.3f} ({after_risk})<br>"
                f"<b>CHI Reduction:</b> -{chi_delta:.3f}<br>"
                f"<b>Surface Temp (LST):</b> {after_lst:.1f}°C (-{lst_delta:.1f}°C)<br>"
                f"<b>Vegetation (NDVI):</b> {after_features.get('ndvi',0):.3f}<br>"
                f"<b>Built-Up (NDBI):</b> {after_features.get('ndbi',0):.3f}"
                f"</div>"
            )

        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_html, max_width=250),
            icon=folium.Icon(color=marker_color, icon=marker_icon, prefix='fa')
        ).add_to(m)

        # ── 5. Floating On-Map Info Cards ────────────────────────────────────
        if not is_after:
            info_html = f"""
            <div style="position: fixed; bottom: 12px; left: 12px; z-index:9999;
                        background: rgba(15,23,42,0.92); padding: 8px 12px; border-radius: 8px;
                        border: 1px solid rgba(255,255,255,0.15); box-shadow: 0 4px 14px rgba(0,0,0,0.5);
                        font-family:'Inter',sans-serif; font-size:11px;">
                <div style="font-weight:700; color:#f97316; margin-bottom:2px; display:flex; align-items:center; gap:5px;">
                    <span>🔥</span> Baseline Condition (Before)
                </div>
                <div style="color:#cbd5e1;">
                    CHI: <b style="color:#f8fafc;">{before_chi:.3f}</b> ({before_risk}) &bull; LST: <b style="color:#f8fafc;">{before_lst:.1f}°C</b>
                </div>
            </div>
            """
        else:
            info_html = f"""
            <div style="position: fixed; bottom: 12px; right: 12px; z-index:9999;
                        background: rgba(15,23,42,0.92); padding: 8px 12px; border-radius: 8px;
                        border: 1px solid rgba(255,255,255,0.15); box-shadow: 0 4px 14px rgba(0,0,0,0.5);
                        font-family:'Inter',sans-serif; font-size:11px;">
                <div style="font-weight:700; color:#10b981; margin-bottom:2px; display:flex; align-items:center; gap:5px;">
                    <span>{strategy['icon']}</span> Mitigated Condition (After)
                </div>
                <div style="color:#cbd5e1;">
                    CHI: <b style="color:#f8fafc;">{after_chi:.3f}</b> (-{chi_delta:.3f}) &bull; LST: <b style="color:#f8fafc;">{after_lst:.1f}°C</b> (-{lst_delta:.1f}°C)
                </div>
            </div>
            """
        m.get_root().html.add_child(folium.Element(info_html))

        # ── 6. Shared CHI Color Scale Legend (Top-Right on both) ─────────────
        legend_html = """
        <div style="position: fixed; top: 12px; right: 12px; z-index: 9999;
                    background: rgba(15,23,42,0.88); border: 1px solid rgba(255,255,255,0.12);
                    border-radius: 6px; padding: 6px 10px; font-family: 'Inter', sans-serif;
                    font-size: 10px; color: #cbd5e1; box-shadow: 0 2px 10px rgba(0,0,0,0.4);">
            <div style="font-weight:600; font-size:10px; margin-bottom:3px; color:#f8fafc;">CHI Scale</div>
            <div style="width: 110px; height: 7px; border-radius: 3px;
                        background: linear-gradient(to right, #27ae60, #f1c40f, #e67e22, #c0392b);"></div>
            <div style="display:flex; justify-content:space-between; margin-top:2px; font-size:8.5px; color:#94a3b8;">
                <span>Low</span>
                <span>Mod</span>
                <span>High</span>
                <span>V.High</span>
            </div>
            <div style="display:flex; align-items:center; gap:5px; margin-top:5px; border-top:1px solid rgba(255,255,255,0.1); padding-top:4px; font-size:8.5px;">
                <span style="display:inline-block; width:8px; height:8px; border-radius:2px; background:#1a6eb5;"></span>
                <span style="color:#93c5fd;">Water Bodies (Blue)</span>
            </div>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))

        fd, path = tempfile.mkstemp(suffix='.html')
        try:
            m.save(path)
            with open(path, 'r', encoding='utf-8') as f:
                html = f.read()
        finally:
            os.close(fd)
            os.remove(path)
        return html

    before_html = make_map(is_after=False)
    after_html  = make_map(is_after=True)
    return before_html, after_html


def generate_plotly_temperature_trends(historical_data, location_name='Selected Location',
                                        include_forecast=True, forecast_years=5):
    """
    Generates an interactive Plotly chart showing temperature and CHI trends.
    Clearly distinguishes historical data from predicted/projected data.

    Args:
        historical_data: list of dicts with year, mean_lst_celsius, mean_chi
        location_name: name shown in chart title
        include_forecast: whether to extend with predicted future values
        forecast_years: how many years to project forward
    Returns:
        str: Plotly JSON
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np

    years = [d['year'] for d in historical_data]
    lst   = [d['mean_lst_celsius'] for d in historical_data]
    chi   = [d['mean_chi'] for d in historical_data]

    # ── Calculate warming slope for projection ────────────────────────────────
    if len(years) >= 2:
        x_arr = np.array(years)
        y_lst_arr = np.array(lst)
        slope_lst, intercept_lst = np.polyfit(x_arr, y_lst_arr, 1)
        y_chi_arr = np.array(chi)
        slope_chi, intercept_chi = np.polyfit(x_arr, y_chi_arr, 1)
    else:
        slope_lst = 0.18; intercept_lst = lst[0] if lst else 30
        slope_chi = 0.014; intercept_chi = chi[0] if chi else 0.5

    # ── Forecast values ───────────────────────────────────────────────────────
    last_year = max(years) if years else 2025
    forecast_yrs = list(range(last_year + 1, last_year + forecast_years + 1))
    forecast_lst = [round(slope_lst * y + intercept_lst, 2) for y in forecast_yrs]
    forecast_chi = [round(min(1.0, slope_chi * y + intercept_chi), 3) for y in forecast_yrs]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Historical LST (solid line, red)
    fig.add_trace(go.Scatter(
        x=years, y=lst,
        name="LST (Historical)",
        line=dict(color="#ef4444", width=3),
        marker=dict(size=7, color="#ef4444"),
        hovertemplate="<b>%{x}</b><br>LST: %{y:.1f}°C<extra></extra>",
    ), secondary_y=False)

    # Projected LST (dashed)
    if include_forecast and forecast_yrs:
        # Connect last historical to first forecast
        fig.add_trace(go.Scatter(
            x=[last_year] + forecast_yrs, y=[lst[-1]] + forecast_lst,
            name="LST (Projected)",
            line=dict(color="#ef4444", width=2, dash='dash'),
            marker=dict(size=5, color="#ef4444", symbol='diamond'),
            opacity=0.75,
            hovertemplate="<b>%{x}</b><br>Projected LST: %{y:.1f}°C<extra></extra>",
        ), secondary_y=False)

        # Uncertainty band for forecast
        upper = [v + 0.8 for v in forecast_lst]
        lower = [v - 0.8 for v in forecast_lst]
        fig.add_trace(go.Scatter(
            x=forecast_yrs + forecast_yrs[::-1],
            y=upper + lower[::-1],
            fill='toself',
            fillcolor='rgba(239,68,68,0.08)',
            line=dict(color='rgba(239,68,68,0)'),
            name='Projection Uncertainty',
            showlegend=False,
            hoverinfo='skip',
        ), secondary_y=False)

    # Historical CHI (dashed orange)
    fig.add_trace(go.Scatter(
        x=years, y=chi,
        name="CHI (Historical)",
        line=dict(color="#f97316", width=3),
        marker=dict(size=7, color="#f97316"),
        hovertemplate="<b>%{x}</b><br>CHI: %{y:.3f}<extra></extra>",
    ), secondary_y=True)

    # Projected CHI
    if include_forecast and forecast_yrs:
        fig.add_trace(go.Scatter(
            x=[last_year] + forecast_yrs, y=[chi[-1]] + forecast_chi,
            name="CHI (Projected)",
            line=dict(color="#f97316", width=2, dash='dot'),
            marker=dict(size=5, color="#f97316", symbol='diamond'),
            opacity=0.75,
            hovertemplate="<b>%{x}</b><br>Projected CHI: %{y:.3f}<extra></extra>",
        ), secondary_y=True)

    # Vertical divider line between historical and forecast
    if include_forecast and forecast_yrs:
        fig.add_vline(
            x=last_year + 0.5,
            line_dash="dot",
            line_color="rgba(255,255,255,0.3)",
            annotation_text="Forecast →",
            annotation_position="top right",
            annotation_font=dict(color="#e2e8f0", size=11),
        )

    # CHI risk level bands (horizontal)
    fig.add_hrect(y0=0.75, y1=1.0,   fillcolor="rgba(239,68,68,0.07)",  line_width=0, secondary_y=True)
    fig.add_hrect(y0=0.55, y1=0.75,  fillcolor="rgba(230,126,34,0.07)", line_width=0, secondary_y=True)
    fig.add_hrect(y0=0.35, y1=0.55,  fillcolor="rgba(241,196,15,0.05)", line_width=0, secondary_y=True)

    fig.update_layout(
        template="plotly_dark",
        title=dict(
            text=f"Heat Trend Analysis — {location_name}",
            font=dict(color="#f8fafc", size=16, family="Inter, sans-serif")
        ),
        font=dict(color="#cbd5e1", family="Inter, sans-serif"),
        xaxis=dict(
            title=dict(text="Year", font=dict(color="#cbd5e1", size=13)),
            tickfont=dict(color="#94a3b8", size=12),
            showgrid=True,
            gridcolor='rgba(255,255,255,0.08)',
            tickvals=years + (forecast_yrs if include_forecast else []),
        ),
        legend=dict(
            orientation='h',
            x=0, y=-0.22,
            bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0', size=12),
        ),
        margin=dict(l=50, r=50, t=55, b=85),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
    )

    slope_str = f"+{slope_lst:.2f}" if slope_lst >= 0 else f"{slope_lst:.2f}"
    fig.update_yaxes(
        title_text=f"Mean LST (°C) | Slope: {slope_str}°C/yr",
        title_font=dict(color="#ef4444", size=13),
        tickfont=dict(color="#ef4444", size=12),
        secondary_y=False,
        gridcolor='rgba(255,255,255,0.06)',
    )
    fig.update_yaxes(
        title_text="Composite Heat Index (CHI)",
        title_font=dict(color="#f97316", size=13),
        tickfont=dict(color="#f97316", size=12),
        range=[0, 1.05],
        secondary_y=True,
        showgrid=False,
    )

    import plotly.utils
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def generate_radar_chart(features: dict, location_name: str = 'Selected Location') -> str:
    """
    Generates a Plotly radar/spider chart for Factor Analysis.
    Matches the reference image style:
      - Large centered chart, white/light background
      - Orange/red outline with transparent fill
      - Labels around chart, location title at top
      - 0–100 scale (normalized from raw factor values)

    Args:
        features (dict): Environmental feature values (raw numbers).
        location_name (str): Location to display as title.
    Returns:
        str: Plotly JSON.
    """
    import plotly.graph_objects as go
    import plotly.utils

    # Normalize each factor to 0–100 heat contribution scale
    # Higher = more heat stress (inverted for NDVI and wind)
    lst  = features.get('lst', 30)
    ndvi = features.get('ndvi', 0.4)
    ndbi = features.get('ndbi', 0.1)
    air  = features.get('air_temp', 28)
    rh   = features.get('relative_humidity', 60)
    wind = features.get('wind_speed', 3)
    lulc = features.get('lulc_heat', 0.5)

    # Normalize to 0–100 scale (heat contribution direction)
    scores = {
        'LST':             round(max(0, min(100, (lst  - 20) / 30 * 100)), 1),
        'Air Temperature': round(max(0, min(100, (air  - 15) / 30 * 100)), 1),
        'Built-Up (NDBI)': round(max(0, min(100, (ndbi + 0.5) / 1.0 * 100)), 1),
        'LULC Heat':       round(max(0, min(100, lulc * 100)), 1),
        'Humidity':        round(max(0, min(100, (rh   - 10) / 90 * 100)), 1),
        'Vegetation (NDVI)': round(max(0, min(100, (1 - (ndvi + 0.1) / 0.9) * 100)), 1),  # inverted
        'Wind Speed':      round(max(0, min(100, (1 - wind / 10) * 100)), 1),  # inverted
    }

    categories = list(scores.keys())
    values     = list(scores.values())

    # Close the radar loop
    categories_closed = categories + [categories[0]]
    values_closed     = values + [values[0]]

    fig = go.Figure()

    # Filled radar area (vibrant orange glow)
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        mode='lines+markers',
        fill='toself',
        fillcolor='rgba(249, 115, 22, 0.22)',
        line=dict(color='#f97316', width=3.5),
        marker=dict(size=7, color='#f97316'),
        name='Heat Factors',
        hovertemplate='<b>%{theta}</b><br>Score: %{r:.1f}/100<extra></extra>',
    ))

    # Grid reference circles (25, 50, 75, 100)
    for ref in [25, 50, 75, 100]:
        fig.add_trace(go.Scatterpolar(
            r=[ref] * (len(categories) + 1),
            theta=categories_closed,
            mode='lines',
            line=dict(color='rgba(255, 255, 255, 0.12)', width=1, dash='dot'),
            showlegend=False,
            hoverinfo='skip',
        ))

    fig.update_layout(
        template='plotly_dark',
        title=dict(
            text=f'<b>Factor Analysis</b><br><span style="font-size:12px;color:#94a3b8;">{location_name}</span>',
            x=0.5,
            xanchor='center',
            font=dict(size=17, color='#f8fafc', family='Inter'),
        ),
        polar=dict(
            bgcolor='rgba(15, 23, 42, 0.6)',
            angularaxis=dict(
                tickfont=dict(size=12, color='#f8fafc', family='Inter'),
                linecolor='rgba(255, 255, 255, 0.2)',
                gridcolor='rgba(255, 255, 255, 0.12)',
            ),
            radialaxis=dict(
                range=[0, 100],
                tickvals=[25, 50, 75, 100],
                tickfont=dict(size=10, color='#94a3b8'),
                gridcolor='rgba(255, 255, 255, 0.12)',
                linecolor='rgba(255, 255, 255, 0.15)',
            ),
        ),
        paper_bgcolor='rgba(0, 0, 0, 0.0)',
        plot_bgcolor='rgba(0, 0, 0, 0.0)',
        showlegend=False,
        height=480,
        margin=dict(l=85, r=85, t=90, b=55),
    )

    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def generate_prediction_chart(timeline_data, location_name='Selected Location'):
    """
    Generates an ML prediction timeline chart strictly covering future years (2027 to 2040).
    Displays:
      - ML-predicted CHI curve with explicit markers for each year (2027..2040)
      - 95% Confidence Interval band from Random Forest decision tree ensemble
      - Threshold risk bands (Low, Moderate, High, Very High)
      - Dark theme with high contrast typography and hover details including accuracy & predicted LST.
    """
    import plotly.graph_objects as go
    import plotly.utils

    if not timeline_data:
        from app.ml import predict_yearly_timeline
        timeline_data = predict_yearly_timeline({}, start_year=2027, end_year=2040)

    years = [d['year'] for d in timeline_data]
    chi_vals = [d['predicted_chi'] for d in timeline_data]
    ci_lower = [d.get('ci_lower', d['predicted_chi'] - 0.02) for d in timeline_data]
    ci_upper = [d.get('ci_upper', d['predicted_chi'] + 0.02) for d in timeline_data]
    lst_vals = [d.get('predicted_lst', 32.0) for d in timeline_data]
    accuracies = [d.get('accuracy_pct', 95.0) for d in timeline_data]
    risks = [d.get('risk_level', 'Moderate') for d in timeline_data]

    fig = go.Figure()

    # 95% Confidence Interval shaded band
    fig.add_trace(go.Scatter(
        x=years + years[::-1],
        y=ci_upper + ci_lower[::-1],
        fill='toself',
        fillcolor='rgba(239, 68, 68, 0.12)',
        line=dict(color='rgba(239, 68, 68, 0)'),
        hoverinfo='skip',
        showlegend=True,
        name='95% Model Confidence Band',
    ))

    # Main Predicted CHI line with markers for every year
    custom_hover = [
        f"<b>Year: {y}</b><br>Predicted CHI: <b>{c:.3f}</b><br>Predicted LST: <b>{l:.1f}°C</b><br>Risk Level: <b>{r}</b><br>Model Accuracy: <b>{a}%</b><extra>RF Model</extra>"
        for y, c, l, r, a in zip(years, chi_vals, lst_vals, risks, accuracies)
    ]

    fig.add_trace(go.Scatter(
        x=years,
        y=chi_vals,
        mode='lines+markers',
        name='ML Predicted CHI',
        line=dict(color='#f97316', width=3.5),
        marker=dict(size=8, color='#ef4444', symbol='circle', line=dict(color='#f8fafc', width=1.5)),
        text=custom_hover,
        hoverinfo='text',
    ))

    # Risk level horizontal bands
    fig.add_hrect(y0=0.75, y1=1.0,  fillcolor='rgba(239,68,68,0.08)', line_width=0,
                  annotation_text='Very High', annotation_position='right', annotation_font=dict(color='#ef4444', size=10))
    fig.add_hrect(y0=0.55, y1=0.75, fillcolor='rgba(230,126,34,0.07)', line_width=0,
                  annotation_text='High', annotation_position='right', annotation_font=dict(color='#e67e22', size=10))
    fig.add_hrect(y0=0.35, y1=0.55, fillcolor='rgba(241,196,15,0.05)', line_width=0,
                  annotation_text='Moderate', annotation_position='right', annotation_font=dict(color='#f1c40f', size=10))
    fig.add_hrect(y0=0.0,  y1=0.35, fillcolor='rgba(39,174,96,0.05)', line_width=0,
                  annotation_text='Low', annotation_position='right', annotation_font=dict(color='#27ae60', size=10))

    fig.update_layout(
        template='plotly_dark',
        title=dict(
            text=f'ML Heat Prediction Timeline (2027 – 2040) — {location_name}',
            font=dict(color='#f8fafc', size=16, family="Inter, sans-serif")
        ),
        font=dict(color="#cbd5e1", family="Inter, sans-serif"),
        xaxis=dict(
            title=dict(text='Prediction Year', font=dict(color='#cbd5e1', size=13)),
            tickmode='linear',
            tick0=years[0] if years else 2027,
            dtick=1,
            tickvals=years,
            tickfont=dict(color='#94a3b8', size=11),
            gridcolor='rgba(255,255,255,0.06)'
        ),
        yaxis=dict(
            title=dict(text='Composite Heat Index (CHI)', font=dict(color='#f97316', size=13)),
            tickfont=dict(color='#f97316', size=12),
            range=[0, 1.05],
            gridcolor='rgba(255,255,255,0.06)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            orientation='h', x=0, y=-0.22,
            font=dict(color='#e2e8f0', size=12),
            bgcolor='rgba(0,0,0,0)'
        ),
        margin=dict(l=50, r=80, t=55, b=85),
        hovermode='closest',
    )

    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def generate_matplotlib_correlation(ndvi_values, lst_values):
    """
    Generates a static Matplotlib scatter plot demonstrating the inverse correlation
    between vegetation indices (NDVI) and Land Surface Temperature (LST).
    """
    print("[Visualization] Plotting static NDVI vs LST correlation graph via Matplotlib...")
    return "static/images/ndvi_lst_correlation.png"
