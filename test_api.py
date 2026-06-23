"""
Test script for the Fraud Detection API
Run this to test all API endpoints
"""

import requests
import json
import numpy as np

BASE_URL = "http://localhost:5000"

def test_status():
    """Test API status endpoint"""
    print("\n🔍 Testing /api/status...")
    try:
        response = requests.get(f"{BASE_URL}/api/status")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {json.dumps(response.json(), indent=2)}")
        return True
    except Exception as e:
        print(f"   Error: {e}")
        return False

def test_models():
    """Test models listing endpoint"""
    print("\n🔍 Testing /api/models...")
    try:
        response = requests.get(f"{BASE_URL}/api/models")
        print(f"   Status: {response.status_code}")
        data = response.json()
        print(f"   Current model: {data.get('current_model')}")
        print(f"   Total models: {data.get('total_models')}")
        return True
    except Exception as e:
        print(f"   Error: {e}")
        return False

def test_predict_sample():
    """Test sample prediction endpoint"""
    print("\n🔍 Testing /api/predict/sample...")
    try:
        response = requests.get(f"{BASE_URL}/api/predict/sample")
        print(f"   Status: {response.status_code}")
        data = response.json()
        print(f"   Prediction: {data.get('prediction')} ({data.get('label')})")
        print(f"   Confidence: {data.get('confidence'):.3f}")
        print(f"   Model used: {data.get('model_used')}")
        return True
    except Exception as e:
        print(f"   Error: {e}")
        return False

def test_predict_post():
    """Test POST prediction endpoint"""
    print("\n🔍 Testing /api/predict (POST)...")
    try:
        # Generate random features (20 features as per config)
        features = np.random.randn(20).tolist()
        
        payload = {
            "features": features,
            "true_label": np.random.choice([0, 1])  # Optional: for drift detection
        }
        
        response = requests.post(
            f"{BASE_URL}/api/predict",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"   Status: {response.status_code}")
        data = response.json()
        
        if "error" in data:
            print(f"   Error: {data['error']}")
            return False
        
        print(f"   Prediction: {data.get('prediction')} ({data.get('label')})")
        print(f"   Confidence: {data.get('confidence'):.3f}")
        print(f"   Model used: {data.get('model_used')}")
        return True
        
    except Exception as e:
        print(f"   Error: {e}")
        return False

def test_dashboard():
    """Test dashboard endpoint"""
    print("\n🔍 Testing /api/dashboard...")
    try:
        response = requests.get(f"{BASE_URL}/api/dashboard")
        print(f"   Status: {response.status_code}")
        data = response.json()
        
        if "files_status" in data:
            status = data["files_status"]
            print(f"   Files loaded: {status.get('loaded_files')}/{status.get('total_files')}")
        
        if "instructions" in data:
            print(f"   Note: {data['instructions']['message']}")
            print(f"   Run: python frauddrift.py to generate data")
        
        return True
    except Exception as e:
        print(f"   Error: {e}")
        return False

def test_drift():
    """Test drift detection endpoint"""
    print("\n🔍 Testing /api/drift...")
    try:
        response = requests.get(f"{BASE_URL}/api/drift")
        print(f"   Status: {response.status_code}")
        data = response.json()
        
        if "status" in data:
            print(f"   Drift status: {data['status'].get('status')}")
            print(f"   Total drifts: {data['status'].get('total_drifts_detected')}")
        
        return True
    except Exception as e:
        print(f"   Error: {e}")
        return False

def test_switch_model():
    """Test model switching endpoint"""
    print("\n🔍 Testing /api/switch...")
    try:
        # First get available models
        models_response = requests.get(f"{BASE_URL}/api/models")
        models_data = models_response.json()
        available_models = models_data.get("models", [])
        
        if not available_models:
            print("   No models available for switching")
            return False
        
        # Try to switch to the second model (if available)
        if len(available_models) > 1:
            target_model = available_models[1]["name"]  # Second model
            
            payload = {
                "model": target_model,
                "reason": "API test switch"
            }
            
            response = requests.post(
                f"{BASE_URL}/api/switch",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            print(f"   Status: {response.status_code}")
            data = response.json()
            
            if data.get("success"):
                print(f"   Successfully switched to: {data.get('current_model')}")
                return True
            else:
                print(f"   Failed to switch: {data.get('error', 'Unknown error')}")
                return False
        else:
            print("   Only one model available, cannot switch")
            return True
            
    except Exception as e:
        print(f"   Error: {e}")
        return False

def run_all_tests():
    """Run all API tests"""
    print("=" * 60)
    print("🚀 FRAUD DETECTION API TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Status", test_status),
        ("Models", test_models),
        ("Predict Sample", test_predict_sample),
        ("Predict POST", test_predict_post),
        ("Dashboard", test_dashboard),
        ("Drift", test_drift),
        ("Switch Model", test_switch_model)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n▶️  Running: {test_name}")
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {test_name:20} {status}")
    
    print(f"\n   Total: {passed}/{total} tests passed")
    
    # Recommendations
    print("\n🔧 RECOMMENDATIONS:")
    if passed < total:
        print("   1. Make sure API server is running: python api_server.py")
        print("   2. Run frauddrift.py to generate data files")
        print("   3. Check if all model files (*.pkl) exist")
        print("   4. Verify Flask and dependencies are installed")
    else:
        print("   🎉 All tests passed! API is working correctly.")
        print("   Next: Create a frontend or use the API for predictions")

if __name__ == "__main__":
    run_all_tests()