"""
Flask API Server for Frontend Integration - UPDATED FIX
Run this to create a local API server
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
from datetime import datetime
import json
import os
import traceback

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Global system components
model_manager = None
drift_detector = None

def load_system_components():
    """Load the trained system components"""
    global model_manager, drift_detector
    
    try:
        # Import and initialize
        from model_manager import ProfessionalModelManager
        from drift_detector import AdvancedDriftDetector
        
        # Load model manager
        model_manager = ProfessionalModelManager(load_existing=True)
        
        # Initialize drift detector
        drift_detector = AdvancedDriftDetector(
            window_size=150,
            accuracy_threshold=0.75,
            confidence_level=0.97
        )
        
        print("✅ System components loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to load system components: {e}")
        traceback.print_exc()
        return False

@app.route('/')
def home():
    """Home page - API documentation"""
    return jsonify({
        "api_name": "Fraud Detection System API",
        "version": "2.0",
        "status": "operational",
        "endpoints": {
            "/api/status": "GET - System status",
            "/api/models": "GET - List available models",
            "/api/predict": "POST - Make fraud prediction",
            "/api/predict/sample": "GET - Test prediction with sample data",
            "/api/drift": "GET - Drift detection status",
            "/api/history": "GET - Prediction history",
            "/api/switch": "POST - Switch model manually",
            "/api/dashboard": "GET - Dashboard data"
        },
        "documentation": "Send POST request to /api/predict with JSON: {'features': [array of numbers]}"
    })

@app.route('/api/status', methods=['GET'])
def api_status():
    """Get system status"""
    if not model_manager:
        return jsonify({"error": "System not initialized"}), 500
    
    status = model_manager.get_system_status()
    return jsonify(status)

@app.route('/api/models', methods=['GET'])
def list_models():
    """List all available models"""
    if not model_manager:
        return jsonify({"error": "System not initialized"}), 500
    
    models = []
    for name, info in model_manager.model_info.items():
        models.append({
            "name": name,
            "description": info["description"],
            "type": info["type"],
            "accuracy": info["performance"]["accuracy"],
            "usage_count": info["performance"]["usage_count"],
            "is_current": name == model_manager.current_model
        })
    
    return jsonify({
        "models": models,
        "current_model": model_manager.current_model,
        "total_models": len(models)
    })

@app.route('/api/predict', methods=['POST', 'GET'])
def predict():
    """Make fraud prediction - accepts both POST and GET for testing"""
    if not model_manager:
        return jsonify({"error": "System not initialized"}), 500
    
    try:
        # Handle GET request (for testing)
        if request.method == 'GET':
            # Generate sample features (20 features as per config)
            sample_features = np.random.randn(20).tolist()
            return jsonify({
                "message": "Send POST request with 'features' array",
                "example_request": {
                    "features": sample_features
                },
                "current_model": model_manager.current_model
            })
        
        # Handle POST request
        data = request.json
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        if 'features' not in data:
            return jsonify({"error": "Features array required"}), 400
        
        features = np.array(data['features'])
        
        # Validate features length
        if len(features) != 20:  # Based on config
            return jsonify({
                "error": f"Expected 20 features, got {len(features)}",
                "current_model": model_manager.current_model
            }), 400
        
        # Make prediction
        result = model_manager.predict(features, return_details=True)
        
        # Update drift detector if true label provided
        if 'true_label' in data:
            drift_detector.add_prediction(
                data['true_label'],
                result['prediction'],
                result['confidence'],
                datetime.now().isoformat()
            )
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route('/api/predict/sample', methods=['GET'])
def predict_sample():
    """Make prediction with sample data (for testing)"""
    if not model_manager:
        return jsonify({"error": "System not initialized"}), 500
    
    try:
        # Generate realistic sample features
        sample_features = np.random.randn(20).tolist()
        
        # Make prediction
        result = model_manager.predict(np.array(sample_features), return_details=True)
        
        # Add sample info
        result["sample_features"] = sample_features
        result["note"] = "This is a sample prediction using random data"
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/drift', methods=['GET'])
def drift_status():
    """Get drift detection status"""
    if not drift_detector:
        return jsonify({"error": "Drift detector not initialized"}), 500
    
    try:
        status = drift_detector.get_system_status()
        recent_metrics = drift_detector.get_realtime_metrics(last_n=20)
        
        return jsonify({
            "status": status,
            "recent_metrics": recent_metrics,
            "total_alerts": len(drift_detector.drift_alerts)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/history', methods=['GET'])
def prediction_history():
    """Get recent prediction history"""
    try:
        # Try to load from generated files
        if os.path.exists('frontend_data.json'):
            with open('frontend_data.json', 'r') as f:
                history = json.load(f)
            
            return jsonify({
                "source": "frontend_data.json",
                "recent_predictions": history.get('predictions', [])[-20:],
                "accuracy_history": history.get('accuracy_history', []),
                "model_switches": history.get('model_switches', []),
                "last_updated": history.get('export_time', 'unknown')
            })
        elif os.path.exists('drift_analysis.json'):
            with open('drift_analysis.json', 'r') as f:
                drift_data = json.load(f)
            
            return jsonify({
                "source": "drift_analysis.json",
                "drift_alerts": drift_data.get('drift_alerts', [])[-10:],
                "total_drifts": drift_data.get('performance_summary', {}).get('total_drifts', 0),
                "average_accuracy": drift_data.get('performance_summary', {}).get('average_accuracy', 0)
            })
        else:
            return jsonify({
                "message": "No history files found. Run frauddrift.py first to generate data.",
                "suggestions": [
                    "Run: python frauddrift.py",
                    "Check if frontend_data.json exists"
                ]
            })
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/switch', methods=['POST'])
def switch_model():
    """Switch to a different model"""
    if not model_manager:
        return jsonify({"error": "System not initialized"}), 500
    
    try:
        data = request.json
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        if 'model' not in data:
            return jsonify({"error": "Model name required"}), 400
        
        success = model_manager.switch_model(
            data['model'],
            data.get('reason', 'Manual switch via API')
        )
        
        if success:
            return jsonify({
                "success": True,
                "message": f"Switched to {data['model']}",
                "current_model": model_manager.current_model,
                "available_models": list(model_manager.models.keys())
            })
        else:
            return jsonify({"error": f"Model '{data['model']}' not found"}), 404
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/dashboard', methods=['GET'])
def dashboard_data():
    """Get dashboard data for frontend"""
    try:
        dashboard_data = {
            "api_status": {
                "name": "Fraud Detection System API",
                "version": "2.0",
                "status": "operational",
                "timestamp": datetime.now().isoformat()
            }
        }
        
        # Add system status if available
        if model_manager:
            dashboard_data["system_status"] = model_manager.get_system_status()
            dashboard_data["available_models"] = list(model_manager.models.keys())
        
        # Add drift status if available
        if drift_detector:
            dashboard_data["drift_status"] = drift_detector.get_system_status()
        
        # Load generated data files
        data_files = {
            'frontend_data': 'frontend_data.json',
            'drift_analysis': 'drift_analysis.json', 
            'model_inventory': 'model_inventory.json',
            'system_report': 'system_report.json'
        }
        
        files_loaded = 0
        for key, filename in data_files.items():
            if os.path.exists(filename):
                try:
                    with open(filename, 'r') as f:
                        dashboard_data[key] = json.load(f)
                    files_loaded += 1
                except Exception as e:
                    dashboard_data[key + "_error"] = f"Failed to load: {str(e)}"
            else:
                dashboard_data[key + "_missing"] = True
        
        # Add file status
        dashboard_data["files_status"] = {
            "total_files": len(data_files),
            "loaded_files": files_loaded,
            "missing_files": len(data_files) - files_loaded
        }
        
        # If no data files found, provide instructions
        if files_loaded == 0:
            dashboard_data["instructions"] = {
                "message": "No data files found. To generate dashboard data:",
                "steps": [
                    "1. Run: python frauddrift.py",
                    "2. Wait for the system to complete (takes 1-2 minutes)",
                    "3. Refresh this page"
                ],
                "expected_files": list(data_files.values())
            }
        
        return jsonify(dashboard_data)
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc(),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/api/docs', methods=['GET'])
def api_documentation():
    """API documentation"""
    docs = {
        "api": "Fraud Detection System API v2.0",
        "description": "Professional fraud detection with concept drift handling",
        "endpoints": [
            {
                "endpoint": "/api/predict",
                "methods": ["POST", "GET"],
                "description": "Make fraud prediction",
                "request": {
                    "method": "POST",
                    "content-type": "application/json",
                    "body": {"features": "array[float] - 20 feature values"}
                },
                "example": {
                    "curl": "curl -X POST http://localhost:5000/api/predict -H 'Content-Type: application/json' -d '{\"features\": [0.1, 0.2, ... 20 values]}'",
                    "python": "requests.post('http://localhost:5000/api/predict', json={'features': [0.1, 0.2, ...]})"
                }
            },
            {
                "endpoint": "/api/predict/sample",
                "method": "GET",
                "description": "Test prediction with sample data",
                "response": "Prediction result with random features"
            },
            {
                "endpoint": "/api/dashboard",
                "method": "GET",
                "description": "Get complete dashboard data",
                "response": "All data needed for frontend dashboard"
            }
        ],
        "quick_start": [
            "1. Run frauddrift.py to generate models and data",
            "2. Start API: python api_server.py",
            "3. Test API: curl http://localhost:5000/api/status",
            "4. Make prediction: curl -X POST http://localhost:5000/api/predict -H 'Content-Type: application/json' -d '{\"features\": [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9, -1.0, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9, -1.0]}'"
        ]
    }
    
    return jsonify(docs)

if __name__ == '__main__':
    # Load system components
    print("🚀 Starting Fraud Detection API Server...")
    print("=" * 60)
    
    if load_system_components():
        print("✅ API Server Ready!")
        print("📡 Endpoints available at:")
        print("   http://localhost:5000/                    - API Documentation")
        print("   http://localhost:5000/api/status          - System Status")
        print("   http://localhost:5000/api/predict         - Make Prediction (POST)")
        print("   http://localhost:5000/api/predict/sample  - Test Prediction (GET)")
        print("   http://localhost:5000/api/dashboard       - Dashboard Data")
        print("   http://localhost:5000/api/docs            - Detailed Documentation")
        print("\n📝 Quick Test Commands:")
        print("   curl http://localhost:5000/api/status")
        print("   curl http://localhost:5000/api/predict/sample")
        print("\n🔧 For full data, first run: python frauddrift.py")
        print("   (This generates the dashboard data files)")
        print("\nPress Ctrl+C to stop the server")
        print("=" * 60)
        
        # Run Flask server
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to start API server")