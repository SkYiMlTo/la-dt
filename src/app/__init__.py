from flask import Flask
from flask_cors import CORS
import threading

def create_app():
    app = Flask(__name__)
    CORS(app)

    from .routes import analysis_bp
    app.register_blueprint(analysis_bp)

    # Start the background data analysis thread
    from .detector import start_detector
    detector_thread = threading.Thread(target=start_detector, daemon=True)
    detector_thread.start()

    return app
