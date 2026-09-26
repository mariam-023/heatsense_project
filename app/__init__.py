import os
# pyrefly: ignore [missing-import]
from flask import Flask
from datetime import timedelta
from app.database import init_db

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def create_app(test_config=None):
    """
    Application Factory Pattern for Flask.
    Creates and configures the HeatSense application instance.
    """
    app = Flask(__name__, instance_relative_config=True)

    # Default configuration
    app.config.from_mapping(
        SECRET_KEY=os.environ.get('SECRET_KEY', 'heatsense_secure_session_key_2026_karnataka'),
        DATABASE=os.path.join(app.instance_path, 'heatsense.sqlite'),
        SESSION_PERMANENT=True,
        PERMANENT_SESSION_LIFETIME=timedelta(hours=24),
        # Firebase configuration (set via environment variables)
        FIREBASE_API_KEY=           os.environ.get('FIREBASE_API_KEY', ''),
        FIREBASE_AUTH_DOMAIN=       os.environ.get('FIREBASE_AUTH_DOMAIN', ''),
        FIREBASE_PROJECT_ID=        os.environ.get('FIREBASE_PROJECT_ID', ''),
        FIREBASE_STORAGE_BUCKET=    os.environ.get('FIREBASE_STORAGE_BUCKET', ''),
        FIREBASE_MESSAGING_SENDER_ID=os.environ.get('FIREBASE_MESSAGING_SENDER_ID', ''),
        FIREBASE_APP_ID=            os.environ.get('FIREBASE_APP_ID', ''),
    )

    if test_config is None:
        app.config.from_pyfile('config.py', silent=True)
    else:
        app.config.from_mapping(test_config)

    # Ensure instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # Initialize the database schema
    with app.app_context():
        init_db(app.config['DATABASE'])

    # Register routes blueprint
    from app.routes import main_bp
    app.register_blueprint(main_bp)

    # Initialize Google Earth Engine in background
    try:
        from app.preprocessing import initialize_earth_engine
        initialize_earth_engine()
    except Exception as e:
        print(f"[Startup] GEE initialization deferred: {e}")

    # Auto-train Random Forest model on first startup
    from app.ml import MODEL_PATH, train_rf_model
    if not os.path.exists(MODEL_PATH):
        print("[Startup] No trained model found. Running training pipeline...")
        with app.app_context():
            try:
                train_rf_model()
            except Exception as e:
                print(f"[Startup] Model training deferred: {e}")

    # Auto-generate today's health alerts for seeded districts at startup
    try:
        from app.health_risk import generate_alerts_for_all_districts
        generate_alerts_for_all_districts(app)
    except Exception as e:
        print(f"[Startup] Health alert generation deferred: {e}")

    return app
