/**
 * Standalone Prophet Forecasting Client & Chart.js Renderer
 */

// Configurable API base URL (can be set via window.PROPHET_API_BASE_URL)
const API_BASE_URL = window.PROPHET_API_BASE_URL || "";

let forecastChart = null;

document.addEventListener("DOMContentLoaded", () => {
  initDashboard();

  document.getElementById("btnRefresh").addEventListener("click", fetchAndRenderForecast);
  document.getElementById("btnRegenerate").addEventListener("click", regenerateForecast);
});

function setStatus(text, isActive = false) {
  const dot = document.getElementById("statusDot");
  const lbl = document.getElementById("statusText");
  if (lbl) lbl.textContent = text;
  if (dot) {
    if (isActive) dot.classList.add("active");
    else dot.classList.remove("active");
  }
}

async function initDashboard() {
  setStatus("Loading forecast...", false);
  await fetchAndRenderForecast();
}

async function fetchAndRenderForecast() {
  try {
    setStatus("Fetching API data...", false);
    const res = await fetch(`${API_BASE_URL}/api/forecast?include_historical=true`);
    if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);

    const data = await res.json();
    if (data.status !== "success") throw new Error(data.message || "Failed to load forecast data");

    renderDashboard(data.historical || [], data.forecast || []);
    setStatus("Connected & Live", true);
  } catch (err) {
    console.error("Failed to load forecast data from API:", err);
    setStatus(`API Error: ${err.message}`, false);
  }
}

async function regenerateForecast() {
  try {
    setStatus("Regenerating Prophet model (this may take a few seconds)...", false);
    const res = await fetch(`${API_BASE_URL}/api/forecast/generate`, { method: "POST" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    alert(`Success: ${data.message}`);
    await fetchAndRenderForecast();
  } catch (err) {
    alert(`Failed to regenerate forecast: ${err.message}`);
    setStatus("Regeneration failed", false);
  }
}

function renderDashboard(historical, forecast) {
  if (!forecast.length) return;

  // 1. Update Metric Cards
  const latestItem = forecast[forecast.length - 1];
  document.getElementById("valLatestTemp").textContent = `${latestItem.yhat.toFixed(1)} °C`;
  document.getElementById("lblLatestDate").textContent = `Forecast Date: ${latestItem.ds}`;

  if (latestItem.yhat_lower !== undefined && latestItem.yhat_upper !== undefined) {
    document.getElementById("valConfidenceRange").textContent = `${latestItem.yhat_lower.toFixed(1)} °C – ${latestItem.yhat_upper.toFixed(1)} °C`;
  }

  if (historical.length) {
    const sYear = historical[0].ds.split("-")[0];
    const eYear = historical[historical.length - 1].ds.split("-")[0];
    document.getElementById("valHistoricalWindow").textContent = `${sYear} – ${eYear}`;
  }

  if (forecast.length) {
    const fStart = forecast[0].ds;
    const fEnd = forecast[forecast.length - 1].ds;
    document.getElementById("lblForecastHorizonSub").textContent = `${fStart} to ${fEnd}`;
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
  const ctx = document.getElementById("forecastChart").getContext("2d");
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
          backgroundColor: "rgba(244, 63, 94, 0.12)",
          borderWidth: 1,
          pointRadius: 0,
          fill: "-1", // Fill area between upper and lower bound
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
          title: { display: true, text: "Temperature (°C)", color: "#94a3b8" }
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
