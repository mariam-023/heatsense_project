"""
Standalone Prophet Forecasting Flask REST API Server
=====================================================
Backend API server that exposes endpoints for retrieving Prophet time-series forecasts,
historical baseline observations, and re-running the forecast pipeline dynamically.
"""

import os
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from flask import Flask, jsonify, request, send_from_directory, Response

# Add current folder to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from prophet_model import (
    load_and_preprocess_historical_data,
    run_prophet_forecast,
    INPUT_TS_FILE,
    FALLBACK_TS_FILE,
    OUTPUT_PREDICTION_FILE,
    DATA_DIR,
    BASE_DIR
)
import pandas as pd

app = Flask(
    __name__,
    static_folder=str(BASE_DIR / "frontend"),
    static_url_path=""
)


@app.after_request
def add_cors_headers(response):
    """Enable CORS for cross-domain frontend integration."""
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response


@app.route("/")
def serve_index():
    """Serve the standalone frontend dashboard."""
    frontend_dir = BASE_DIR / "frontend"
    if (frontend_dir / "index.html").exists():
        return send_from_directory(str(frontend_dir), "index.html")
    return jsonify({
        "status": "online",
        "service": "Prophet Forecasting API",
        "documentation": "Visit /api/forecast for forecast data or /api/health for system status."
    })


@app.route("/<path:path>")
def serve_static(path):
    """Serve static frontend assets (js, css, images)."""
    frontend_dir = BASE_DIR / "frontend"
    if (frontend_dir / path).exists():
        return send_from_directory(str(frontend_dir), path)
    if (DATA_DIR / path).exists():
        return send_from_directory(str(DATA_DIR), path)
    return send_from_directory(str(frontend_dir), "index.html")


@app.route("/api/health", methods=["GET"])
def health_check():
    """System health check endpoint."""
    input_file = INPUT_TS_FILE if INPUT_TS_FILE.exists() else FALLBACK_TS_FILE
    return jsonify({
        "status": "healthy",
        "service": "Prophet Forecasting Engine",
        "input_dataset_exists": input_file.exists(),
        "input_dataset_path": str(input_file.name),
        "output_prediction_exists": OUTPUT_PREDICTION_FILE.exists(),
        "output_prediction_path": str(OUTPUT_PREDICTION_FILE.name)
    })


@app.route("/api/forecast", methods=["GET"])
def get_forecast():
    """
    Get Prophet forecast and historical observations.
    Query params:
      - format: 'json' (default) or 'csv'
      - include_historical: 'true' (default) or 'false'
    """
    try:
        req_format = request.args.get("format", "json").lower()
        include_hist = request.args.get("include_historical", "true").lower() == "true"

        # Check prediction output file
        if not OUTPUT_PREDICTION_FILE.exists():
            print("Prediction file missing. Running Prophet forecast pipeline...")
            df_hist = load_and_preprocess_historical_data()
            run_prophet_forecast(df_hist)

        # Read predictions CSV
        df_pred = pd.read_csv(OUTPUT_PREDICTION_FILE)

        # If CSV requested, return raw prediction CSV
        if req_format == "csv":
            csv_str = df_pred.to_csv(index=False)
            return Response(csv_str, mimetype="text/csv", headers={"Content-Disposition": "attachment;filename=prediction.csv"})

        # Read historical data if requested
        historical_records = []
        if include_hist:
            try:
                df_hist = load_and_preprocess_historical_data()
                historical_records = [
                    {"ds": row["ds"].strftime("%Y-%m-%d"), "temp": round(float(row["y"]), 2)}
                    for _, row in df_hist.iterrows()
                ]
            except Exception as e:
                print(f"Warning: Failed to parse historical series: {e}")

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

        return jsonify({
            "status": "success",
            "historical_count": len(historical_records),
            "forecast_count": len(forecast_records),
            "historical": historical_records,
            "forecast": forecast_records
        })

    except Exception as e:
        print(f"Error in /api/forecast: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/forecast/generate", methods=["POST"])
def generate_forecast_endpoint():
    """Re-run the Prophet model training & prediction pipeline on demand."""
    try:
        periods = int(request.args.get("periods", 3650))
        df_hist = load_and_preprocess_historical_data()
        forecast_future, _ = run_prophet_forecast(df_hist, periods=periods)
        return jsonify({
            "status": "success",
            "message": f"Successfully generated {len(forecast_future)} prediction steps.",
            "forecast_count": len(forecast_future),
            "start_date": str(forecast_future["ds"].min()),
            "end_date": str(forecast_future["ds"].max())
        })
    except Exception as e:
        print(f"Error generating forecast: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Standalone Prophet Forecasting Server on http://127.0.0.1:{port}...")
    app.run(host="127.0.0.1", port=port, debug=False)
