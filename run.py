import os
from app import create_app

# Create Flask application instance using the factory pattern
app = create_app()

if __name__ == '__main__':
    # Retrieve configuration from environment or use default development values
    host = os.environ.get('FLASK_RUN_HOST', '0.0.0.0')
    port = int(os.environ.get('FLASK_RUN_PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'True').lower() in ('true', '1', 't')

    print(f"Starting HeatSense Application Server at http://{host}:{port}/")
    app.run(host=host, port=port, debug=debug)
