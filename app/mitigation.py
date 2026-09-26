"""
HeatSense Mitigation Simulation Module — Module 6.

Simulates four urban cooling strategies by adjusting environmental
factor values and re-running the trained Random Forest model to
predict the post-mitigation Composite Heat Index (CHI).

Strategies
----------
1. vegetation_expansion  — Increase vegetation cover (+NDVI, -LULC heat)
2. green_roofs           — Green roof retrofit  (+NDVI, -LST, -LULC heat)
3. reduce_buildup        — Reduce built-up density (-NDBI, -LULC heat, -LST)
4. increase_parks        — Add urban parks (+NDVI, -NDBI, -AirT, -LULC heat)
"""

import json
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Strategy Definitions
# Each strategy specifies absolute offsets applied to the feature vector.
# Negative = cooling / greening;  Positive = warming / hardening.
# ──────────────────────────────────────────────────────────────────────────────

STRATEGIES = {
    'vegetation_expansion': {
        'label':       'Vegetation Expansion',
        'icon':        '🌿',
        'description': 'Increase vegetation cover across urban zones by planting '
                       'street trees, roadside greenery, and urban forests.',
        'colour':      '#27ae60',
        'offsets': {
            'ndvi':      +0.18,   # significant greening
            'ndbi':      -0.04,   # slight reduction in built-up signal
            'lulc_heat': -0.12,   # land-use becomes cooler
            'lst':       -1.8,    # surface cooling via transpiration
        }
    },
    'green_roofs': {
        'label':       'Green Roof Retrofit',
        'icon':        '🏗️',
        'description': 'Install vegetated rooftop gardens on commercial and '
                       'residential buildings to reduce rooftop heat absorption.',
        'colour':      '#2ecc71',
        'offsets': {
            'ndvi':      +0.10,
            'lulc_heat': -0.08,
            'lst':       -1.2,    # rooftop surface temperature reduction
            'air_temp':  -0.5,    # slight ambient cooling
        }
    },
    'reduce_buildup': {
        'label':       'Reduce Built-Up Density',
        'icon':        '🏙️',
        'description': 'Enforce building density limits and convert impervious '
                       'surfaces to permeable pavements and open spaces.',
        'colour':      '#e67e22',
        'offsets': {
            'ndbi':      -0.15,   # major reduction in built-up index
            'lulc_heat': -0.16,
            'lst':       -2.0,    # reduced heat retention from concrete
            'air_temp':  -0.8,
        }
    },
    'increase_parks': {
        'label':       'Urban Park Expansion',
        'icon':        '🌳',
        'description': 'Develop new urban parks and green corridors that create '
                       'cool islands, increase biodiversity, and improve air quality.',
        'colour':      '#16a085',
        'offsets': {
            'ndvi':      +0.22,   # strongest vegetation increase
            'ndbi':      -0.08,
            'lulc_heat': -0.14,
            'lst':       -2.2,    # strongest surface cooling
            'air_temp':  -1.0,    # largest ambient temperature drop
        }
    },
}

# Legacy strategy key mapping (for backwards-compat with old JS 'greenery'/'albedo' values)
LEGACY_KEY_MAP = {
    'greenery': 'vegetation_expansion',
    'albedo':   'green_roofs',
}


# ──────────────────────────────────────────────────────────────────────────────
# Core Simulation Engine
# ──────────────────────────────────────────────────────────────────────────────

def apply_strategy_offsets(features: dict, strategy_key: str) -> dict:
    """
    Applies the strategy's factor offsets to a copy of the feature dict.
    Clamps each feature to its physically valid range.

    Args:
        features     (dict): Current district feature values.
        strategy_key (str) : Key from STRATEGIES.
    Returns:
        dict: Modified feature values after applying offsets.
    """
    CLAMP_RANGES = {
        'lst':               (10.0, 55.0),
        'ndvi':              (-0.1,  0.9),
        'ndbi':              (-0.6,  0.6),
        'air_temp':          (10.0, 50.0),
        'relative_humidity': (5.0, 100.0),
        'wind_speed':        (0.0,  15.0),
        'lulc_heat':         (0.0,   1.0),
    }
    # Resolve legacy keys
    strategy_key = LEGACY_KEY_MAP.get(strategy_key, strategy_key)
    strategy = STRATEGIES.get(strategy_key, {})
    offsets  = strategy.get('offsets', {})

    modified = dict(features)
    for feat, delta in offsets.items():
        if feat in modified:
            lo, hi = CLAMP_RANGES.get(feat, (-1e9, 1e9))
            modified[feat] = round(float(np.clip(modified[feat] + delta, lo, hi)), 4)

    return modified


def run_mitigation_simulation(district_features: dict, strategy_key: str) -> dict:
    """
    Full mitigation simulation pipeline:
      1. Record 'before' feature values and predict before-CHI.
      2. Apply strategy offsets.
      3. Predict after-CHI.
      4. Compute temperature difference and percentage improvement.
      5. Generate comparison Plotly charts.

    Args:
        district_features (dict): Current environmental feature values.
        strategy_key      (str) : One of the four STRATEGIES keys.
    Returns:
        dict: Full simulation result payload.
    """
    from app.ml import predict_current_conditions, chi_to_risk_level, \
        FEATURE_LABELS, FEATURE_COLOURS

    # Resolve legacy keys
    resolved_key = LEGACY_KEY_MAP.get(strategy_key, strategy_key)

    strategy = STRATEGIES.get(resolved_key)
    if not strategy:
        raise ValueError(f"Unknown strategy key: '{strategy_key}'. "
                         f"Valid keys: {list(STRATEGIES.keys())}")

    # ── Before ────────────────────────────────────────────────────────────────
    before_features = dict(district_features)
    before_chi      = predict_current_conditions(before_features)
    before_risk     = chi_to_risk_level(before_chi)
    before_lst      = before_features.get('lst', 32.0)

    # ── Apply strategy ────────────────────────────────────────────────────────
    after_features = apply_strategy_offsets(before_features, resolved_key)
    after_chi      = predict_current_conditions(after_features)
    after_risk     = chi_to_risk_level(after_chi)
    after_lst      = after_features.get('lst', before_lst)

    # ── Delta calculations ────────────────────────────────────────────────────
    chi_reduction   = round(before_chi - after_chi, 4)
    lst_reduction   = round(before_lst - after_lst,  2)
    pct_improvement = round((chi_reduction / before_chi * 100)
                             if before_chi > 0 else 0.0, 2)

    # ── Build factor-delta summary ────────────────────────────────────────────
    offsets = strategy.get('offsets', {})
    factor_changes = []
    for feat, delta in offsets.items():
        factor_changes.append({
            'feature': feat,
            'label':   FEATURE_LABELS.get(feat, feat),
            'colour':  FEATURE_COLOURS.get(feat, '#888'),
            'before':  round(before_features.get(feat, 0), 4),
            'after':   round(after_features.get(feat, 0), 4),
            'delta':   round(delta, 4),
        })

    # ── Generate comparison charts ────────────────────────────────────────────
    charts = _build_comparison_charts(
        before_chi, after_chi,
        before_risk, after_risk,
        before_lst, after_lst,
        factor_changes, strategy
    )

    return {
        'status': 'success',
        'strategy_key':   resolved_key,
        'strategy_label': strategy['label'],
        'strategy_icon':  strategy['icon'],
        'description':    strategy['description'],
        'colour':         strategy['colour'],
        # Before
        'before_chi':  before_chi,
        'before_risk': before_risk,
        'before_lst':  round(before_lst, 2),
        # After
        'after_chi':  after_chi,
        'after_risk': after_risk,
        'after_lst':  round(after_lst, 2),
        # Differences
        'chi_reduction':         chi_reduction,
        'lst_reduction':         lst_reduction,
        'lst_reduction_celsius': lst_reduction,
        'pct_improvement':       pct_improvement,
        # Legacy field kept for any existing JS references
        'simulated_lst_reduction': lst_reduction,
        # Detail
        'factor_changes': factor_changes,
        # Plotly chart JSONs
        **charts,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Chart Generation
# ──────────────────────────────────────────────────────────────────────────────

def _build_comparison_charts(
    before_chi, after_chi,
    before_risk, after_risk,
    before_lst, after_lst,
    factor_changes, strategy
) -> dict:
    """
    Builds four Plotly charts for the before/after comparison panel:
      1. CHI grouped bar — before / after side-by-side
      2. LST overlay bar — surface temp reduction
      3. Factor-delta bar — which factors changed and by how much
      4. Improvement gauge/indicator — post-mitigation CHI with delta

    Returns:
        dict with keys: chi_bar_json, lst_bar_json, delta_bar_json, indicator_json
    """
    # pyrefly: ignore [missing-import]
    import plotly.graph_objects as go
    # pyrefly: ignore [missing-import]
    import plotly.utils

    colour_before = '#e74c3c'
    colour_after  = strategy['colour']

    # ── 1. CHI Before / After grouped bar ────────────────────────────────────
    chi_bar = go.Figure(data=[
        go.Bar(name='Before Mitigation', x=['Composite Heat Index'],
               y=[before_chi], marker_color=colour_before,
               text=[f'{before_chi:.3f}'], textposition='outside'),
        go.Bar(name='After Mitigation',  x=['Composite Heat Index'],
               y=[after_chi],  marker_color=colour_after,
               text=[f'{after_chi:.3f}'], textposition='outside'),
    ])
    chi_bar.update_layout(
        title=f'{strategy["icon"]} CHI Before vs After — {strategy["label"]}',
        barmode='group', height=300,
        yaxis=dict(range=[0, 1.15], title='CHI Value'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=30, r=20, t=50, b=50),
        legend=dict(orientation='h', y=-0.3),
    )

    # ── 2. LST Before / After overlay bar ────────────────────────────────────
    lst_bar = go.Figure(data=[
        go.Bar(name='Before', y=['Surface Temp (LST)'],
               x=[before_lst], orientation='h', marker_color=colour_before,
               text=[f'{before_lst:.1f}°C'], textposition='outside'),
        go.Bar(name='After',  y=['Surface Temp (LST)'],
               x=[after_lst],  orientation='h', marker_color=colour_after,
               text=[f'{after_lst:.1f}°C'], textposition='outside'),
    ])
    lst_bar.update_layout(
        title='Surface Temperature Reduction (°C)',
        barmode='overlay', height=200,
        xaxis=dict(range=[0, max(before_lst, after_lst) * 1.18],
                   title='Temperature (°C)'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=70, t=50, b=50),
        legend=dict(orientation='h', y=-0.5),
    )

    # ── 3. Factor-delta horizontal bar ────────────────────────────────────────
    f_labels = [f['label']  for f in factor_changes]
    f_deltas = [f['delta']  for f in factor_changes]
    f_colors = [f['colour'] for f in factor_changes]
    f_texts  = [f'+{d:.3f}' if d > 0 else f'{d:.3f}' for d in f_deltas]

    delta_bar = go.Figure(go.Bar(
        y=f_labels, x=f_deltas,
        orientation='h',
        marker_color=f_colors,
        text=f_texts, textposition='outside',
        hovertemplate='<b>%{y}</b><br>Change: %{x:.4f}<extra></extra>',
    ))
    delta_bar.update_layout(
        title='Environmental Factor Changes Applied',
        height=300,
        xaxis=dict(title='Delta', zeroline=True,
                   zerolinecolor='#555', zerolinewidth=1.5),
        yaxis=dict(showgrid=False),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=70, t=50, b=30),
    )

    # ── 4. Improvement indicator (gauge) ─────────────────────────────────────
    pct_drop = abs(before_chi - after_chi) / before_chi * 100 if before_chi > 0 else 0
    indicator = go.Figure(go.Indicator(
        mode='number+delta+gauge',
        value=after_chi,
        delta=dict(reference=before_chi,
                   decreasing=dict(color=colour_after),
                   increasing=dict(color=colour_before),
                   valueformat='.3f'),
        gauge=dict(
            axis=dict(range=[0, 1]),
            bar=dict(color=colour_after),
            steps=[
                dict(range=[0.00, 0.35], color='#d5f5e3'),
                dict(range=[0.35, 0.55], color='#fef9e7'),
                dict(range=[0.55, 0.75], color='#fdebd0'),
                dict(range=[0.75, 1.00], color='#fadbd8'),
            ],
            threshold=dict(line=dict(color=colour_before, width=3),
                           thickness=0.85, value=before_chi),
        ),
        number=dict(suffix='  CHI', valueformat='.3f'),
        title=dict(text=(
            f'Post-Mitigation CHI — {strategy["label"]}<br>'
            f'<span style="font-size:0.78em">'
            f'Improvement: {abs(before_chi - after_chi):.3f} '
            f'({pct_drop:.1f}% reduction)'
            f'</span>'
        )),
    ))
    indicator.update_layout(
        height=300,
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=30, r=30, t=70, b=20),
    )

    def _j(fig):
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return {
        'chi_bar_json':   _j(chi_bar),
        'lst_bar_json':   _j(lst_bar),
        'delta_bar_json': _j(delta_bar),
        'indicator_json': _j(indicator),
    }
