from flask import Flask, jsonify, request
import os
from flask_cors import CORS

def create_app():
    app = Flask(__name__, 
                template_folder='../frontend/templates',
                static_folder='../frontend/static')

    # Configure CORS with specific settings
    CORS(app, resources={
        r"/analyze": {"origins": "*"},
        r"/search": {"origins": "*"},
        r"/predict": {"origins": "*"},
        r"/export": {"origins": "*"},
        r"/download/*": {"origins": "*"}
    })

    app.config.update(
        EXPORT_FOLDER='exports',
        SECRET_KEY='tata-motors-material-analyzer-2025'
    )

    os.makedirs(app.config['EXPORT_FOLDER'], exist_ok=True)

    # Flexible import - try both ways
    try:
        # Try absolute import first (for when run from parent directory)
        from server.routes import main
    except ImportError:
        # Fall back to relative import (for when run from server directory)
        from routes import main
    
    app.register_blueprint(main)

    # Fixed error handlers - return JSON instead of trying to render templates
    @app.errorhandler(404)
    def not_found(error):
        # Always return JSON for API endpoints
        return jsonify({
            'success': False, 
            'error': 'Endpoint not found',
            'message': 'The requested URL was not found on the server'
        }), 404

    @app.errorhandler(500)
    def internal_error(error):
        # Always return JSON for API endpoints  
        return jsonify({
            'success': False, 
            'error': 'Internal server error',
            'message': 'An internal server error occurred'
        }), 500

    # Add a test route to verify the app is working
    @app.route('/test')
    def test():
        return jsonify({'success': True, 'message': 'App is working'})

    # Add a simple health check route
    @app.route('/health')
    def health():
        return jsonify({'status': 'healthy', 'service': 'Material Analyzer'})

    return app

# For debugging - add this if you're running the app directly
if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)