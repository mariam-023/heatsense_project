/**
 * HeatSense — Main Dashboard JavaScript
 * Handles page navigation, API data loading, chart rendering,
 * mitigation simulation, before/after swipe, and age-specific alerts.
 */

(function () {
    'use strict';

    // ── Global State ──────────────────────────────────────────────────────
    const HS = window.HEATSENSE || { lat: 12.9716, lon: 77.5946, name: 'Bengaluru Urban', radius: 15 };
    let currentPage = 'home';
    let selectedStrategy = 'vegetation_expansion';
    let predictionData = null;
    let factorData = null;
    let healthData = null;

    // ── Page Navigation ───────────────────────────────────────────────────
    document.addEventListener('DOMContentLoaded', () => {
        initNavigation();
        // Load home data on startup
        loadPrediction();
        loadHealthRisk();
    });

    function initNavigation() {
        const navLinks = document.querySelectorAll('#dashboard-nav .nav-link');
        navLinks.forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const page = link.dataset.page;
                switchPage(page);
            });
        });
    }

    function switchPage(page) {
        // Hide all pages
        document.querySelectorAll('.dashboard-page').forEach(p => p.classList.remove('active'));
        document.querySelectorAll('#dashboard-nav .nav-link').forEach(l => l.classList.remove('active'));

        // Show target page
        const target = document.getElementById('page-' + page);
        if (target) {
            target.classList.add('active');
            currentPage = page;
        }

        // Highlight nav link
        const navLink = document.querySelector(`#dashboard-nav .nav-link[data-page="${page}"]`);
        if (navLink) navLink.classList.add('active');

        if (page === 'prediction') {
            initProphetButtons();
            fetchAndRenderForecast();
        }
        if (page === 'factor-analysis') loadFactorAnalysis();
        if (page === 'before-after') {
            const beforeIframe = document.getElementById('ba-before-iframe');
            if (!beforeIframe || !beforeIframe.srcdoc) {
                loadBeforeAfter();
            }
        }
        if (page === 'health-risk') loadHealthRisk();
        if (page === 'alerts-precautions') {
            loadAlertsPrecautions();
            if (typeof loadSmsStatusIdx === 'function') loadSmsStatusIdx();
        }
    }

    // ── API Helper ────────────────────────────────────────────────────────
    function apiUrl(path) {
        const sep = path.includes('?') ? '&' : '?';
        return `${path}${sep}lat=${HS.lat}&lon=${HS.lon}&name=${encodeURIComponent(HS.name)}&radius=${HS.radius}`;
    }

    function fetchJSON(url) {
        return fetch(url).then(r => {
            if (!r.ok) throw new Error(`HTTP ${r.status}`);
            return r.json();
        });
    }

    // ── HOME: Load Predictions (populates home stats + prediction page) ──
    function loadPrediction() {
        fetchJSON(apiUrl('/api/predict'))
            .then(data => {
                predictionData = data;
                updateHomeDashboard(data);
                updatePredictionPage(data);
                // Automatically fetch and render Prophet forecast matching active location
                fetchAndRenderForecast();
            })
            .catch(err => {
                console.error('Prediction load failed:', err);
                setEl('home-stat-chi', 'Error');
            });
    }

    function updateHomeDashboard(data) {
        setEl('home-stat-chi', data.current_chi.toFixed(3));
        setEl('home-stat-lst', data.predicted_lst_celsius.toFixed(1) + '°C');
        setEl('home-overall-temp', (data.overall_temperature_celsius || data.predicted_lst_celsius).toFixed(1) + '°C');
        setEl('home-feels-like', (data.feels_like_celsius || data.overall_temperature_celsius || data.predicted_lst_celsius).toFixed(1) + '°C');
        setEl('home-stat-risk', data.current_risk_level);
        setEl('home-stat-advisory', data.current_advisory);

        // Color risk level
        const riskEl = document.getElementById('home-stat-risk');
        if (riskEl) riskEl.style.color = riskColor(data.current_risk_level);

        // Environmental factors
        if (data.features) {
            setEl('home-air-temp', (data.features.air_temp || 0).toFixed(1) + '°C');
            setEl('home-humidity', (data.features.relative_humidity || 0).toFixed(1) + '%');
            setEl('home-ndvi', (data.features.ndvi || 0).toFixed(3));
            setEl('home-ndbi', (data.features.ndbi || 0).toFixed(3));
            setEl('home-wind',  (data.features.wind_speed || 0).toFixed(1) + ' m/s');
            setEl('home-wind2', (data.features.wind_speed || 0).toFixed(1) + ' m/s');
            setEl('home-lulc', (data.features.lulc_heat || 0).toFixed(3));
        }

        setEl('home-future-chi', data.future_chi.toFixed(3));
        setEl('home-data-source', 'Data source: ' + (data.data_source === 'gee' ? '🛰️ Google Earth Engine' : '📊 Climatological Model'));

        // Map GEE badge
        const mapBadge = document.getElementById('map-gee-badge');
        const mapText = document.getElementById('map-gee-text');
        if (mapBadge && mapText) {
            if (data.data_source === 'gee') {
                mapBadge.className = 'gee-badge online';
                mapText.textContent = 'GEE Layers';
            } else {
                mapBadge.className = 'gee-badge offline';
                mapText.textContent = 'Basemap';
            }
        }

        // Alert banner for high/very high
        if (data.current_risk_level === 'High' || data.current_risk_level === 'Very High') {
            const alertDiv = document.getElementById('dashboard-alert');
            if (alertDiv) {
                alertDiv.classList.remove('d-none');
                setEl('dashboard-alert-text', data.current_advisory);
            }
        }
    }

    function updatePredictionPage(data) {
        setEl('pred-lst', data.predicted_lst_celsius.toFixed(1) + '°C');
        setEl('pred-chi', data.current_chi.toFixed(3));
        setEl('pred-risk', data.current_risk_level);
        setEl('pred-future-chi', data.future_chi.toFixed(3));
        setEl('pred-future-risk', data.future_risk_level);
        setEl('pred-advisory', data.future_advisory);

        const riskEl = document.getElementById('pred-risk');
        if (riskEl) riskEl.style.color = riskColor(data.current_risk_level);
        const futRiskEl = document.getElementById('pred-future-risk');
        if (futRiskEl) futRiskEl.style.color = riskColor(data.future_risk_level);
    }

    // ── TRENDS ────────────────────────────────────────────────────────────
    let trendsLoaded = false;
    window.loadTrends = function () {
        const startYear = document.getElementById('trend-start-year')?.value || 2015;
        const endYear = document.getElementById('trend-end-year')?.value || 2025;

        const chartDiv = document.getElementById('trend-chart');
        chartDiv.innerHTML = '<div class="loading-overlay"><div class="hs-spinner"></div><span>Loading trend analysis...</span></div>';

        fetchJSON(apiUrl(`/api/history?start_year=${startYear}&end_year=${endYear}`))
            .then(data => {
                if (data.chart && data.chart.data) {
                    const layout = data.chart.layout || {};
                    layout.template = 'plotly_dark';
                    layout.paper_bgcolor = 'rgba(0,0,0,0)';
                    layout.plot_bgcolor  = 'rgba(0,0,0,0)';
                    layout.font = { color: '#cbd5e1', family: 'Inter, sans-serif' };
                    if (layout.title) {
                        if (typeof layout.title === 'string') {
                            layout.title = { text: layout.title, font: { color: '#f8fafc', size: 16 } };
                        } else {
                            layout.title.font = { color: '#f8fafc', size: 16 };
                        }
                    }
                    if (layout.xaxis) {
                        layout.xaxis.title = typeof layout.xaxis.title === 'string'
                            ? { text: layout.xaxis.title, font: { color: '#cbd5e1' } }
                            : { ...(layout.xaxis.title || {}), font: { color: '#cbd5e1' } };
                        layout.xaxis.tickfont = { color: '#94a3b8' };
                    }
                    if (layout.legend) {
                        layout.legend.font = { color: '#e2e8f0' };
                    }
                    Plotly.newPlot('trend-chart', data.chart.data, layout, {
                        responsive: true,
                        displayModeBar: true,
                        displaylogo: false,
                    });
                }
                // Update slope info
                if (data.data && data.data.length > 0) {
                    const last = data.data[data.data.length - 1];
                    setEl('trend-slope-info',
                        `Warming rate: +${last.lst_slope.toFixed(3)}°C/year | ` +
                        `Latest LST: ${last.mean_lst_celsius.toFixed(1)}°C | ` +
                        `Latest CHI: ${last.mean_chi.toFixed(3)} | ` +
                        `Data source: ${last.data_source === 'gee' ? 'Google Earth Engine' : 'Climatological Model'}`);
                }
                trendsLoaded = true;
            })
            .catch(err => {
                console.error('Trends failed:', err);
                chartDiv.innerHTML = '<div class="loading-overlay text-danger">Failed to load trend data.</div>';
            });
    };

    // ── PROPHET TIME-SERIES FORECASTING ───────────────────────────────────
    let forecastChart = null;
    let isFetchingForecast = false;
    let pendingForecastPromise = null;
    let isProphetBaselineMode = false;
    let latestForecastRequestId = 0;

    function initProphetButtons() {
        const btnRefresh = document.getElementById('btnRefresh');
        const btnRegenerate = document.getElementById('btnRegenerate');

        if (btnRefresh && !btnRefresh.dataset.bound) {
            btnRefresh.dataset.bound = "true";
            btnRefresh.addEventListener('click', () => fetchAndRenderForecast(true));
        }

        if (btnRegenerate && !btnRegenerate.dataset.bound) {
            btnRegenerate.dataset.bound = "true";
            btnRegenerate.addEventListener('click', regenerateForecast);
        }
    }

    window.toggleProphetBaseline = function () {
        isProphetBaselineMode = !isProphetBaselineMode;
        const btn = document.getElementById('btn-prophet-toggle-baseline');
        const lbl = document.getElementById('lbl-prophet-baseline-toggle');
        if (btn && lbl) {
            if (isProphetBaselineMode) {
                btn.className = "btn btn-sm btn-warning text-dark fw-bold";
                lbl.textContent = "Switch to Active Location";
            } else {
                btn.className = "btn btn-sm btn-outline-warning fw-semibold";
                lbl.textContent = "Statewide Regional Baseline";
            }
        }
        fetchAndRenderForecast(true);
    };

    function setProphetStatus(text, isActive = false) {
        const lbl = document.getElementById('statusText');
        const badge = document.getElementById('statusDotBadge');
        if (lbl) lbl.textContent = text;
        if (badge) {
            badge.className = isActive ? 'gee-badge online' : 'gee-badge offline';
        }
    }

    window.fetchAndRenderForecast = async function (forceRefresh = false) {
        const requestId = ++latestForecastRequestId;

        const overlay = document.getElementById('prophet-loading-overlay');
        const errDiv = document.getElementById('prophet-error-state');
        const errMsg = document.getElementById('prophet-error-msg');
        const noticeBanner = document.getElementById('prophet-notice-banner');
        const noticeText = document.getElementById('prophet-notice-text');
        const scopeBadge = document.getElementById('prophet-scope-badge');
        const locBadge = document.getElementById('prophet-location-badge');

        if (locBadge) {
            if (isProphetBaselineMode) {
                locBadge.textContent = '🏛️ Karnataka Statewide Baseline';
                locBadge.className = 'badge bg-warning text-dark fw-bold';
            } else {
                locBadge.textContent = `📍 Active Location: ${HS.name || 'Selected Location'}`;
                locBadge.className = 'badge bg-warning bg-opacity-10 text-warning border border-warning border-opacity-25';
            }
        }

        // Only show full loading overlay if chart has never been rendered
        if (!forecastChart && overlay) {
            overlay.classList.remove('d-none');
        }
        if (errDiv) errDiv.classList.add('d-none');

        isFetchingForecast = true;
        try {
            setProphetStatus(forecastChart ? "Refreshing forecast..." : "Fetching Prophet forecast API...", false);

            let url = `/api/forecast?include_historical=true`;
            if (isProphetBaselineMode) {
                url += `&baseline=true`;
            } else {
                url += `&lat=${HS.lat}&lon=${HS.lon}&name=${encodeURIComponent(HS.name || '')}`;
                if (HS.district_id) url += `&district_id=${HS.district_id}`;
            }

            const res = await fetch(url);

            let data = null;
            try {
                data = await res.json();
            } catch (jErr) {
                throw new Error(`HTTP ${res.status}: Server returned invalid response`);
            }

            // Discard stale out-of-order response
            if (requestId !== latestForecastRequestId) {
                console.log(`[Prophet] Ignoring out-of-order response #${requestId} (latest is #${latestForecastRequestId})`);
                return;
            }

            if (!res.ok || data.status !== "success") {
                throw new Error(data.message || `HTTP ${res.status}: ${res.statusText}`);
            }

            // Update notice banner
            if (noticeBanner && noticeText) {
                if (data.notice && data.notice.trim().length > 0) {
                    noticeText.textContent = data.notice;
                    noticeBanner.classList.remove('d-none');
                } else {
                    noticeBanner.classList.add('d-none');
                }
            }

            // Update scope badge
            if (scopeBadge) {
                const scopeText = data.spatial_unit || 'Spatial Unit';
                if (data.data_scope === 'district_extracted') {
                    scopeBadge.textContent = `Scope: District Satellite (${data.location_label})`;
                    scopeBadge.className = 'badge bg-success-subtle text-success border border-success';
                } else if (data.data_scope === 'point_extracted') {
                    scopeBadge.textContent = `Scope: Local MODIS Pixel (${data.location_label})`;
                    scopeBadge.className = 'badge bg-info-subtle text-info border border-info';
                } else if (data.data_scope === 'regional_fallback') {
                    scopeBadge.textContent = `Scope: Regional Baseline (Fallback for ${data.requested_location || 'Selected Area'})`;
                    scopeBadge.className = 'badge bg-warning-subtle text-warning border border-warning';
                } else {
                    scopeBadge.textContent = `Scope: ${data.location_label || 'Karnataka Statewide (Regional Baseline)'}`;
                    scopeBadge.className = 'badge bg-secondary-subtle text-light border border-secondary';
                }
            }

            renderProphetDashboard(data.historical || [], data.forecast || [], data.location_label);
            setProphetStatus("Connected & Live", true);
            if (overlay) overlay.classList.add('d-none');
        } catch (err) {
            if (requestId !== latestForecastRequestId) return;
            console.error("[Prophet Dashboard] Forecast load failed:", err);
            setProphetStatus(`API Error: ${err.message}`, false);
            if (overlay) overlay.classList.add('d-none');

            // Keep existing rendered chart visible if present; only show full-card error if no chart exists
            if (!forecastChart && errDiv) {
                errDiv.classList.remove('d-none');
                if (errMsg) errMsg.textContent = `API Error: ${err.message}`;
            }
        } finally {
            if (requestId === latestForecastRequestId) {
                isFetchingForecast = false;
                pendingForecastPromise = null;
            }
        }
    };

    async function regenerateForecast() {
        const btnRegen = document.getElementById('btnRegenerate');
        let url = `/api/forecast/generate?lat=${HS.lat}&lon=${HS.lon}&name=${encodeURIComponent(HS.name || '')}`;
        if (isProphetBaselineMode) {
            url = `/api/forecast/generate?location=regional`;
        }

        if (btnRegen) {
            btnRegen.disabled = true;
            btnRegen.innerHTML = '<i class="fa-solid fa-spinner fa-spin me-1"></i> Regenerating...';
        }

        try {
            setProphetStatus(`Retraining Prophet model for '${HS.name || 'Location'}'...`, false);
            const res = await fetch(url, { method: "POST" });
            const data = await res.json();
            if (!res.ok || data.status !== "success") {
                throw new Error(data.message || `HTTP ${res.status}`);
            }
            alert(`Success: ${data.message}`);
            await fetchAndRenderForecast(true);
        } catch (err) {
            alert(`Failed to regenerate forecast: ${err.message}`);
            setProphetStatus("Regeneration failed", false);
        } finally {
            if (btnRegen) {
                btnRegen.disabled = false;
                btnRegen.innerHTML = '<i class="fa-solid fa-bolt me-1"></i> Re-run Prophet Model';
            }
        }
    }

    function renderProphetDashboard(historical, forecast) {
        if (!forecast || !forecast.length) return;

        // 1. Update Metric Cards
        const latestItem = forecast[forecast.length - 1];
        setEl('valLatestTemp', `${latestItem.yhat.toFixed(1)} °C`);
        setEl('lblLatestDate', `Forecast Date: ${latestItem.ds}`);

        if (latestItem.yhat_lower !== undefined && latestItem.yhat_upper !== undefined) {
            setEl('valConfidenceRange', `${latestItem.yhat_lower.toFixed(1)} °C – ${latestItem.yhat_upper.toFixed(1)} °C`);
        }

        if (historical.length) {
            const sYear = historical[0].ds.split("-")[0];
            const eYear = historical[historical.length - 1].ds.split("-")[0];
            setEl('valHistoricalWindow', `${sYear} – ${eYear}`);
        }

        if (forecast.length) {
            const fStart = forecast[0].ds;
            const fEnd = forecast[forecast.length - 1].ds;
            setEl('lblForecastHorizonSub', `${fStart} to ${fEnd}`);
        }

        // 2. Downsample for chart performance (~150 max points)
        const downHist = downsampleSeries(historical, 150);
        const downForecast = downsampleSeries(forecast, 150);

        // 3. Assemble combined labels and series
        const labels = downHist.map(d => d.ds).concat(downForecast.map(d => d.ds));
        const histSeries = downHist.map(d => d.temp).concat(new Array(downForecast.length).fill(null));
        const forecastSeries = new Array(downHist.length).fill(null).concat(downForecast.map(d => d.yhat));
        const lowerSeries = new Array(downHist.length).fill(null).concat(downForecast.map(d => d.yhat_lower ?? d.yhat));
        const upperSeries = new Array(downHist.length).fill(null).concat(downForecast.map(d => d.yhat_upper ?? d.yhat));

        // 4. Render Chart.js
        const canvas = document.getElementById("forecastChart");
        if (!canvas) return;
        const ctx = canvas.getContext("2d");
        if (forecastChart) forecastChart.destroy();

        forecastChart = new Chart(ctx, {
            type: "line",
            data: {
                labels: labels,
                datasets: [
                    {
                        label: "Historical LST (°C)",
                        data: histSeries,
                        borderColor: "#38bdf8",
                        backgroundColor: "#38bdf8",
                        borderWidth: 2,
                        pointRadius: 0,
                        pointHoverRadius: 4,
                        spanGaps: true
                    },
                    {
                        label: "Prophet Forecast (°C)",
                        data: forecastSeries,
                        borderColor: "#f43f5e",
                        backgroundColor: "#f43f5e",
                        borderWidth: 2,
                        pointRadius: 0,
                        pointHoverRadius: 4,
                        spanGaps: true
                    },
                    {
                        label: "Upper Confidence Bound (°C)",
                        data: upperSeries,
                        borderColor: "rgba(244, 63, 94, 0.25)",
                        borderWidth: 1,
                        pointRadius: 0,
                        fill: false,
                        spanGaps: true
                    },
                    {
                        label: "Lower Confidence Bound (°C)",
                        data: lowerSeries,
                        borderColor: "rgba(244, 63, 94, 0.25)",
                        backgroundColor: "rgba(244, 63, 94, 0.15)",
                        borderWidth: 1,
                        pointRadius: 0,
                        fill: "-1",
                        spanGaps: true
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: "index",
                    intersect: false
                },
                plugins: {
                    legend: {
                        labels: {
                            color: "#94a3b8",
                            filter: item => !item.text.includes("Confidence Bound")
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: context => `${context.dataset.label}: ${context.parsed.y !== null ? context.parsed.y.toFixed(2) + ' °C' : 'N/A'}`
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { color: "rgba(255, 255, 255, 0.05)" },
                        ticks: { color: "#94a3b8", maxTicksLimit: 12 }
                    },
                    y: {
                        grid: { color: "rgba(255, 255, 255, 0.05)" },
                        ticks: { color: "#94a3b8" },
                        title: { display: true, text: "Land Surface Temperature (°C)", color: "#94a3b8" }
                    }
                }
            }
        });
    }

    function downsampleSeries(series, maxPoints) {
        if (!series || series.length <= maxPoints) return series;
        const result = [];
        const step = Math.max(1, Math.floor(series.length / maxPoints));
        for (let i = 0; i < series.length; i += step) {
            result.push(series[i]);
        }
        if (result[result.length - 1] !== series[series.length - 1]) {
            result.push(series[series.length - 1]);
        }
        return result;
    }

    let predChartLoaded = false;
    function loadPredictionChart() {
        initProphetButtons();
        fetchAndRenderForecast();
        predChartLoaded = true;
    }

    // ── FACTOR ANALYSIS ───────────────────────────────────────────────────
    let factorLoaded = false;
    function loadFactorAnalysis() {
        if (factorLoaded) return;

        // Helper to clear loading spinner from a container and return it
        function clearLoader(id) {
            const el = document.getElementById(id);
            if (el) {
                // Remove any loading-overlay children
                el.querySelectorAll('.loading-overlay').forEach(o => o.remove());
            }
            return el;
        }

        fetchJSON(apiUrl('/api/factor-analysis'))
            .then(data => {
                factorData = data;

                // ── Radar chart ──────────────────────────────────────────
                const radarEl = clearLoader('radar-chart');
                if (radarEl && data.radar_chart_json) {
                    try {
                        const radarData = JSON.parse(data.radar_chart_json);
                        radarData.layout = radarData.layout || {};
                        radarData.layout.paper_bgcolor = 'rgba(0,0,0,0)';
                        radarData.layout.plot_bgcolor  = 'rgba(0,0,0,0)';
                        if (radarData.layout.polar) {
                            radarData.layout.polar.bgcolor = 'rgba(15, 23, 42, 0.45)';
                        }
                        Plotly.newPlot('radar-chart', radarData.data, radarData.layout, {
                            responsive: true,
                            displaylogo: false,
                            displayModeBar: false,
                        });
                    } catch (e) {
                        console.error('Radar chart render error:', e);
                        radarEl.innerHTML = '<div class="loading-overlay" style="color:#ef4444;">⚠️ Chart render failed. Refresh and try again.</div>';
                    }
                }

                // ── Factor score cards ───────────────────────────────────
                clearLoader('factor-scores-container');
                if (data.factor_scores && data.factor_scores.length > 0) {
                    renderFactorCards(data.factor_scores);
                }

                // ── Bar chart (RF feature importance) ───────────────────
                if (data.bar_chart_json) {
                    try {
                        const barData = JSON.parse(data.bar_chart_json);
                        barData.layout = barData.layout || {};
                        barData.layout.paper_bgcolor = 'rgba(0,0,0,0)';
                        barData.layout.plot_bgcolor  = 'rgba(0,0,0,0)';
                        barData.layout.font = { color: '#cbd5e1' };
                        Plotly.newPlot('factor-bar-chart', barData.data, barData.layout, { responsive: true, displaylogo: false });
                    } catch (e) { console.error('Bar chart error:', e); }
                }

                // ── Pie chart ────────────────────────────────────────────
                if (data.pie_chart_json) {
                    try {
                        const pieData = JSON.parse(data.pie_chart_json);
                        pieData.layout = pieData.layout || {};
                        pieData.layout.paper_bgcolor = 'rgba(0,0,0,0)';
                        pieData.layout.plot_bgcolor  = 'rgba(0,0,0,0)';
                        pieData.layout.font = { color: '#cbd5e1' };
                        Plotly.newPlot('factor-pie-chart', pieData.data, pieData.layout, { responsive: true, displaylogo: false });
                    } catch (e) { console.error('Pie chart error:', e); }
                }

                // ── Key findings text ────────────────────────────────────
                if (data.factor_scores && data.factor_scores.length > 0) {
                    const sorted = [...data.factor_scores].sort((a, b) => b.score - a.score);
                    const top3   = sorted.slice(0, 3).map(f => f.label).join(', ');
                    setEl('factor-findings',
                        `The top contributing heat factors for ${HS.name} are: ${top3}. ` +
                        `Data source: ${data.data_source === 'gee' ? 'Google Earth Engine satellite imagery' : 'Climatological regression model'}.`
                    );
                }

                factorLoaded = true;
            })
            .catch(err => {
                console.error('Factor analysis failed:', err);
                const radarEl = document.getElementById('radar-chart');
                if (radarEl) radarEl.innerHTML = '<div class="loading-overlay" style="color:#ef4444;">⚠️ Failed to load factor data. Check your connection.</div>';
                clearLoader('factor-scores-container');
            });
    }

    function renderFactorCards(factors) {
        const container = document.getElementById('factor-scores-container');
        container.innerHTML = factors.map(f => `
            <div class="factor-card mb-2">
                <span class="factor-icon">${f.icon}</span>
                <div class="flex-grow-1">
                    <div class="d-flex justify-content-between">
                        <span class="factor-label">${f.label}</span>
                        <span class="factor-value" style="color:${f.color};">${f.value}${f.unit ? ' ' + f.unit : ''}</span>
                    </div>
                    <div class="factor-bar">
                        <div class="factor-bar-fill" style="width:${f.score}%;background:${f.color};"></div>
                    </div>
                    <div class="d-flex justify-content-between mt-1">
                        <small style="font-size:0.65rem;color:var(--text-muted);">${f.direction === 'heat' ? '🔥 Heat factor' : '❄️ Cooling factor'}</small>
                        <small style="font-size:0.65rem;color:var(--text-muted);">${f.score}/100</small>
                    </div>
                </div>
            </div>
        `).join('');
    }

    // ── MITIGATION ────────────────────────────────────────────────────────
    window.selectStrategy = function (el) {
        document.querySelectorAll('.strategy-card').forEach(c => c.classList.remove('selected'));
        el.classList.add('selected');
        selectedStrategy = el.dataset.strategy;
    };

    window.runMitigation = function () {
        const btn = document.getElementById('btn-run-mitigation');
        btn.innerHTML = '<div class="hs-spinner" style="width:16px;height:16px;border-width:2px;"></div> Simulating...';
        btn.disabled = true;

        fetch('/api/mitigation', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                lat: HS.lat,
                lon: HS.lon,
                name: HS.name,
                radius: HS.radius,
                scenario_type: selectedStrategy,
            }),
        })
            .then(r => r.json())
            .then(data => {
                btn.innerHTML = '<i class="fa-solid fa-play me-1"></i> Run Simulation';
                btn.disabled = false;

                document.getElementById('mitigation-results').classList.remove('d-none');

                setEl('mit-before-chi', Number(data.before_chi || 0).toFixed(3));
                setEl('mit-after-chi', Number(data.after_chi || 0).toFixed(3));

                const lstReduction = Number(data.lst_reduction_celsius ?? data.lst_reduction ?? data.simulated_lst_reduction ?? 0);
                setEl('mit-lst-reduction', '-' + lstReduction.toFixed(1) + '°C');

                const beforeChi = Number(data.before_chi || 0);
                const afterChi = Number(data.after_chi || 0);
                const pct = beforeChi > 0 ? ((beforeChi - afterChi) / beforeChi * 100).toFixed(1) : '0.0';
                setEl('mit-pct', '-' + pct + '%');

                // Before/After comparison chart
                const chiTrace = {
                    x: ['Before Mitigation', 'After Mitigation'],
                    y: [beforeChi, afterChi],
                    type: 'bar',
                    marker: {
                        color: ['#ef4444', '#10b981'],
                        line: { width: 0 },
                    },
                    text: [beforeChi.toFixed(3), afterChi.toFixed(3)],
                    textposition: 'outside',
                    textfont: { color: '#f8fafc', size: 14, family: 'Inter, sans-serif' },
                };
                Plotly.newPlot('mit-chi-chart', [chiTrace], {
                    title: {
                        text: '<b>CHI Before vs After Mitigation</b>',
                        font: { color: '#f8fafc', size: 15, family: 'Inter, sans-serif' }
                    },
                    yaxis: {
                        range: [0, 1],
                        gridcolor: 'rgba(255,255,255,0.08)',
                        tickfont: { color: '#94a3b8' },
                        title: { text: 'Composite Heat Index', font: { color: '#cbd5e1' } }
                    },
                    xaxis: {
                        tickfont: { color: '#f8fafc', size: 13 }
                    },
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor: 'rgba(0,0,0,0)',
                    margin: { t: 50, b: 40, l: 50, r: 20 },
                }, { responsive: true, displaylogo: false });

                // Gauge chart
                const gaugeTrace = {
                    type: 'indicator',
                    mode: 'gauge+number+delta',
                    value: afterChi,
                    delta: { reference: beforeChi, decreasing: { color: '#10b981' } },
                    number: { font: { color: '#f8fafc', size: 28 }, valueformat: '.3f' },
                    gauge: {
                        axis: { range: [0, 1], tickfont: { color: '#94a3b8' } },
                        bar: { color: '#10b981' },
                        steps: [
                            { range: [0, 0.35], color: 'rgba(39,174,96,0.3)' },
                            { range: [0.35, 0.55], color: 'rgba(241,196,15,0.3)' },
                            { range: [0.55, 0.75], color: 'rgba(230,126,34,0.3)' },
                            { range: [0.75, 1], color: 'rgba(239,68,68,0.3)' },
                        ],
                        threshold: {
                            line: { color: '#ef4444', width: 3 },
                            thickness: 0.8,
                            value: beforeChi,
                        },
                    },
                    title: { text: '<b>Post-Mitigation Heat Index</b>', font: { color: '#f8fafc', size: 15 } },
                };
                Plotly.newPlot('mit-gauge-chart', [gaugeTrace], {
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor: 'rgba(0,0,0,0)',
                    margin: { t: 50, b: 20, l: 30, r: 30 },
                    font: { color: '#f8fafc', family: 'Inter, sans-serif' },
                }, { responsive: true, displaylogo: false });
            })
            .catch(err => {
                console.error('Mitigation failed:', err);
                btn.innerHTML = '<i class="fa-solid fa-play me-1"></i> Run Simulation';
                btn.disabled = false;
            });
    };

    // ── BEFORE & AFTER ────────────────────────────────────────────────────
    window.loadBeforeAfter = function () {
        const strategy = document.getElementById('ba-strategy')?.value || 'vegetation_expansion';
        const loading = document.getElementById('ba-loading');
        const stats = document.getElementById('ba-stats');

        // Show loading overlay, hide stats
        loading.style.display = 'flex';
        loading.innerHTML = '<div class="hs-spinner"></div><span>Loading Before &amp; After maps...</span>';
        stats.style.display = 'none';

        // Reset slider to center position on each new load
        const slider = document.getElementById('ba-slider');
        const beforeDiv = document.getElementById('ba-before');
        if (slider) slider.style.left = '50%';
        if (beforeDiv) beforeDiv.style.clipPath = 'inset(0 50% 0 0)';

        // Fire both API calls in parallel
        const mapsPromise = fetchJSON(apiUrl(`/api/before-after-maps?strategy=${strategy}`));
        const mitPromise  = fetch('/api/mitigation', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ lat: HS.lat, lon: HS.lon, name: HS.name, radius: HS.radius, scenario_type: strategy }),
        }).then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); });

        Promise.all([mapsPromise, mitPromise])
            .then(([mapData, mitData]) => {
                // ── Inject map HTML into iframes ──────────────────────────
                const beforeIframe = document.getElementById('ba-before-iframe');
                const afterIframe  = document.getElementById('ba-after-iframe');

                // Track how many iframes have finished loading
                let iframesLoaded = 0;
                function onIframeLoad() {
                    iframesLoaded++;
                    if (iframesLoaded >= 2) {
                        // Both iframes rendered — hide the loading overlay
                        loading.style.display = 'none';
                    }
                }

                beforeIframe.onload = onIframeLoad;
                afterIframe.onload  = onIframeLoad;

                // Setting srcdoc triggers iframe load
                beforeIframe.srcdoc = mapData.before_html;
                afterIframe.srcdoc  = mapData.after_html;

                // Safety fallback: hide overlay after 8 s even if onload is delayed
                setTimeout(() => { loading.style.display = 'none'; }, 8000);

                // ── Populate stats cards ──────────────────────────────────
                stats.style.display = 'block';

                const beforeChi = Number(mitData.before_chi || 0);
                const afterChi  = Number(mitData.after_chi  || 0);
                const beforeLst = Number(mitData.before_lst || 32.0);
                const afterLst  = Number(mitData.after_lst  || (beforeLst - 1.8));

                const chiDelta   = (beforeChi - afterChi).toFixed(3);
                const lstDelta   = Number(mitData.lst_reduction_celsius ?? mitData.lst_reduction ?? mitData.simulated_lst_reduction ?? (beforeLst - afterLst)).toFixed(1);
                const pctImprove = Number(mitData.pct_improvement || ((beforeChi - afterChi) / Math.max(beforeChi, 0.001) * 100)).toFixed(1);

                // Primary comparison cards
                setEl('ba-chi-before', beforeChi.toFixed(3));
                setEl('ba-chi-after',  afterChi.toFixed(3));
                setEl('ba-chi-delta',  `-${chiDelta} (${pctImprove}% drop)`);

                setEl('ba-lst-before', beforeLst.toFixed(1) + '°C');
                setEl('ba-lst-after',  afterLst.toFixed(1)  + '°C');
                setEl('ba-lst-delta',  `-${lstDelta}°C Cooling`);

                // Risk level badges
                const riskBeforeEl = document.getElementById('ba-risk-before');
                const riskAfterEl  = document.getElementById('ba-risk-after');
                const riskStatusEl = document.getElementById('ba-risk-status');

                const getRiskBadgeClass = (lvl) => {
                    if (lvl === 'Low')      return 'bg-success';
                    if (lvl === 'Moderate') return 'bg-warning text-dark';
                    if (lvl === 'High')     return 'bg-danger';
                    return 'bg-danger';
                };

                if (riskBeforeEl) {
                    riskBeforeEl.textContent = mitData.before_risk || 'High';
                    riskBeforeEl.className   = 'badge ' + getRiskBadgeClass(mitData.before_risk);
                }
                if (riskAfterEl) {
                    riskAfterEl.textContent = mitData.after_risk || 'Moderate';
                    riskAfterEl.className   = 'badge ' + getRiskBadgeClass(mitData.after_risk);
                }
                if (riskStatusEl) {
                    if (mitData.before_risk !== mitData.after_risk) {
                        riskStatusEl.innerHTML = `<span class="text-success"><i class="fa-solid fa-arrow-down me-1"></i>De-escalated to ${mitData.after_risk}</span>`;
                    } else {
                        riskStatusEl.innerHTML = `<span class="text-info"><i class="fa-solid fa-check me-1"></i>Risk reduced within ${mitData.after_risk} tier</span>`;
                    }
                }

                // Strategy card
                setEl('ba-strat-name', (mitData.strategy_icon || '🌿') + ' ' + (mitData.strategy_label || 'Vegetation Expansion'));
                setEl('ba-strat-desc', mitData.description || 'Targeted cooling strategy applied across urban zone.');

                // Environmental Factors Breakdown Table
                const tbody = document.getElementById('ba-factors-tbody');
                if (tbody) {
                    const factorBenefits = {
                        'ndvi':      'Canopy shading and microclimate transpiration cooling',
                        'ndbi':      'Reduction in impervious heat-trapping concrete surfaces',
                        'lst':       'Direct surface thermal cooling via vegetative albedo & shade',
                        'air_temp':  'Reduced ambient air temperature in near-surface boundary layer',
                        'lulc_heat': 'Conversion of thermal-storing surfaces into green cooling space',
                    };
                    const factorIcons = {
                        'ndvi':      '🌿',
                        'ndbi':      '🏙️',
                        'lst':       '🌡️',
                        'air_temp':  '🌬️',
                        'lulc_heat': '🗺️',
                    };

                    let factorList = mitData.factor_changes || [];
                    if (!factorList.length) {
                        factorList = [
                            { feature: 'lst',      label: 'Land Surface Temp (LST)',  before: beforeLst,     after: afterLst,     delta: -Number(lstDelta) },
                            { feature: 'ndvi',     label: 'Vegetation Index (NDVI)',   before: 0.38,          after: 0.56,         delta: 0.18 },
                            { feature: 'ndbi',     label: 'Built-Up Area (NDBI)',      before: 0.12,          after: 0.08,         delta: -0.04 },
                            { feature: 'air_temp', label: 'Air Temperature',           before: beforeLst - 3, after: afterLst - 3, delta: -0.8 },
                        ];
                    }

                    tbody.innerHTML = factorList.map(f => {
                        const icon    = factorIcons[f.feature] || '📊';
                        const unit    = (f.feature === 'lst' || f.feature === 'air_temp') ? '°C' : '';
                        const isGood  = (f.feature === 'ndvi' && f.delta > 0) || (f.feature !== 'ndvi' && f.delta < 0);
                        const cls     = isGood ? 'badge bg-success-subtle text-success' : 'badge bg-secondary-subtle text-muted';
                        const sign    = f.delta > 0 ? '+' : '';
                        const dp      = f.feature === 'lst' ? 1 : 3;
                        const delta   = `${sign}${Number(f.delta).toFixed(dp)}${unit}`;
                        const benefit = factorBenefits[f.feature] || 'Microclimate heat mitigation improvement';
                        return `
                        <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
                            <td><span class="me-2">${icon}</span><b>${f.label}</b></td>
                            <td style="text-align:center;color:#ef4444;font-weight:600;">${Number(f.before).toFixed(dp)}${unit}</td>
                            <td style="text-align:center;color:#10b981;font-weight:600;">${Number(f.after).toFixed(dp)}${unit}</td>
                            <td style="text-align:center;"><span class="${cls}" style="font-size:0.8rem;padding:4px 8px;">${delta}</span></td>
                            <td style="color:#cbd5e1;font-size:0.8rem;">${benefit}</td>
                        </tr>`;
                    }).join('');
                }
            })
            .catch(err => {
                console.error('Before/After failed:', err);
                loading.style.display = 'flex';
                loading.innerHTML = '<span class="text-danger"><i class="fa-solid fa-circle-exclamation me-2"></i>Failed to load comparison. Please try again.</span>';
            });
    };

    // Initialize Before/After swipe slider
    function initSwipeSlider() {
        const wrapper = document.getElementById('ba-wrapper');
        const slider = document.getElementById('ba-slider');
        const before = document.getElementById('ba-before');
        if (!wrapper || !slider || !before) return;

        let isDragging = false;

        function setPosition(clientX) {
            const rect = wrapper.getBoundingClientRect();
            if (rect.width <= 0) return;
            let pct = ((clientX - rect.left) / rect.width) * 100;
            pct = Math.max(2, Math.min(98, pct));
            slider.style.left = pct + '%';
            before.style.clipPath = `inset(0 ${100 - pct}% 0 0)`;
        }

        function startDrag(e) {
            isDragging = true;
            wrapper.querySelectorAll('iframe').forEach(f => { f.style.pointerEvents = 'none'; });
            const clientX = e.touches ? e.touches[0].clientX : e.clientX;
            setPosition(clientX);
            if (e.cancelable) e.preventDefault();
        }

        function stopDrag() {
            if (isDragging) {
                isDragging = false;
                wrapper.querySelectorAll('iframe').forEach(f => { f.style.pointerEvents = ''; });
            }
        }

        function onDrag(e) {
            if (!isDragging) return;
            const clientX = e.touches ? e.touches[0].clientX : e.clientX;
            setPosition(clientX);
            if (e.cancelable) e.preventDefault();
        }

        slider.addEventListener('mousedown', startDrag);
        slider.addEventListener('touchstart', startDrag, { passive: false });

        window.addEventListener('mouseup', stopDrag);
        window.addEventListener('touchend', stopDrag);
        window.addEventListener('touchcancel', stopDrag);

        window.addEventListener('mousemove', onDrag);
        window.addEventListener('touchmove', onDrag, { passive: false });
    }
    document.addEventListener('DOMContentLoaded', initSwipeSlider);

    // ── HEALTH RISK ───────────────────────────────────────────────────────
    function loadHealthRisk() {
        fetchJSON(apiUrl('/api/health-risk'))
            .then(data => {
                healthData = data;
                setEl('hr-hhri', data.hhri.toFixed(3));
                setEl('hr-chi', data.chi.toFixed(3));
                setEl('hr-advisory', data.advisory);

                const hhriEl = document.getElementById('hr-hhri');
                if (hhriEl) hhriEl.style.color = riskColor(data.risk_level);

                const badgeDiv = document.getElementById('hr-risk-badge');
                if (badgeDiv) {
                    const cls = data.risk_level.toLowerCase().replace(' ', '');
                    badgeDiv.innerHTML = `<span class="risk-badge ${cls}">${data.risk_icon} ${data.risk_level} — ${data.action}</span>`;
                }
            })
            .catch(err => console.error('Health risk failed:', err));
    }

    // ── ALERTS & AGE PRECAUTIONS ──────────────────────────────────────────
    let alertsLoaded = false;
    function loadAlertsPrecautions() {
        if (alertsLoaded) return;

        // Load age-specific precautions
        fetchJSON(apiUrl('/api/health-risk'))
            .then(data => {
                const container = document.getElementById('age-cards-container');
                if (data.age_precautions && data.age_precautions.length > 0) {
                    container.innerHTML = data.age_precautions.map(group => `
                        <div class="col-md-4 col-lg-4 mb-3">
                            <div class="age-card ${group.key}">
                                <div class="age-icon">${group.icon}</div>
                                <div class="age-title" style="color:${group.color};">${group.label}</div>
                                <ul class="precaution-list">
                                    ${group.precautions.map(p => `<li>${p}</li>`).join('')}
                                </ul>
                            </div>
                        </div>
                    `).join('');
                } else {
                    container.innerHTML = '<p class="text-muted">No precaution data available.</p>';
                }
            })
            .catch(err => {
                console.error('Precautions failed:', err);
                document.getElementById('age-cards-container').innerHTML = '<p class="text-danger">Failed to load precautions.</p>';
            });

        // Load system alerts
        fetchJSON('/api/alerts')
            .then(alerts => {
                const container = document.getElementById('system-alerts-container');
                if (alerts.length === 0) {
                    container.innerHTML = '<p class="text-muted">No active alerts.</p>';
                    return;
                }
                container.innerHTML = alerts.slice(0, 10).map(a => {
                    const cls = a.risk_level === 'Very High' ? 'danger' : a.risk_level === 'High' ? 'warning' : 'secondary';
                    return `
                    <div class="alert-toast mb-2">
                        <span class="fs-4">${a.risk_level === 'Very High' ? '🔴' : a.risk_level === 'High' ? '🟠' : '🟡'}</span>
                        <div class="flex-grow-1">
                            <div class="d-flex justify-content-between align-items-center mb-1">
                                <strong style="font-size:0.875rem;color:#f1f5f9;">${a.district_name}</strong>
                                <span class="badge bg-${cls}" style="font-size:0.7rem;">${a.risk_level}</span>
                            </div>
                            <div style="font-size:0.8rem;color:#cbd5e1;line-height:1.45;margin-bottom:0.3rem;">${a.advisory_message.substring(0, 150)}...</div>
                            <div style="font-size:0.7rem;color:#94a3b8;">📅 ${a.alert_date} &nbsp;|&nbsp; 🔔 ${a.status}</div>
                        </div>
                    </div>`;
                }).join('');
                alertsLoaded = true;
            })
            .catch(err => {
                console.error('Alerts fetch failed:', err);
                document.getElementById('system-alerts-container').innerHTML = '<p class="text-danger">Failed to load alerts.</p>';
            });

        alertsLoaded = true;
    }

    // ── REPORTS ───────────────────────────────────────────────────────────
    window.downloadReport = function (type) {
        const params = `lat=${HS.lat}&lon=${HS.lon}&name=${encodeURIComponent(HS.name)}&radius=${HS.radius}`;
        if (type === 'pdf') {
            window.open(`/api/download-pdf?${params}`, '_blank');
        } else {
            window.location.href = `/api/download-report?${params}`;
        }
    };

    // ── MAP REFRESH ───────────────────────────────────────────────────────
    window.refreshMap = function () {
        const mainIframe = document.getElementById('main-map-iframe');
        if (mainIframe) {
            mainIframe.src = apiUrl('/api/map-layers?mode=full');
        }
        const homeIframe = document.getElementById('home-map-iframe');
        if (homeIframe) {
            homeIframe.src = apiUrl('/api/map-layers?mode=home');
        }
    };

    // ── UTILITY ───────────────────────────────────────────────────────────
    function setEl(id, value) {
        const el = document.getElementById(id);
        if (el) el.textContent = value;
    }

    function riskColor(level) {
        const colors = {
            'Low': '#10b981',
            'Moderate': '#eab308',
            'High': '#e67e22',
            'Very High': '#ef4444',
        };
        return colors[level] || '#f5f5f5';
    }

})();
