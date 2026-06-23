"""
Professional Model Management System with REST API support
"""

import joblib
import numpy as np
import json
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ProfessionalModelManager:
    def __init__(self, X_train=None, y_train=None, load_existing=True):
        """
        Professional model manager with API support and persistence
        """
        self.models = {}
        self.model_performance = {}
        self.current_model = "primary"
        self.X_train = X_train
        self.y_train = y_train
        self.model_info = {}
        self.switch_history = []
        self.api_endpoints = []
        
        print("🚀 Initializing Professional Model Management System...")
        
        if load_existing:
            self.load_existing_models()
        
        if X_train is not None and y_train is not None:
            self._initialize_models()
        
        print("✅ Model Management System Ready!")
    
    def load_existing_models(self):
        """Load existing trained models from disk"""
        model_files = [
            'primary_model.pkl', 'precision_specialist_model.pkl',
            'recall_specialist_model.pkl', 'complex_specialist_model.pkl',
            'drift_recovery_model.pkl'
        ]
        
        loaded_count = 0
        for model_file in model_files:
            try:
                model_name = model_file.replace('_model.pkl', '')
                model = joblib.load(model_file)
                self.models[model_name] = model
                loaded_count += 1
                print(f"   ✅ Loaded: {model_name}")
            except:
                continue
        
        if loaded_count > 0:
            print(f"   📁 {loaded_count} models loaded from disk")
    
    def _initialize_models(self):
        """Initialize and train professional models"""
        
        print("\n🎯 Training Professional Fraud Detection Models...")
        
        # Model 1: Primary Model
        print("1. 🎯 Primary Model (Balanced Logistic Regression)...")
        model1 = LogisticRegression(
            random_state=42, 
            max_iter=2000, 
            C=1.0, 
            class_weight='balanced',
            solver='lbfgs'
        )
        model1.fit(self.X_train, self.y_train)
        self.models["primary"] = model1
        self._initialize_model_info("primary", "Primary balanced classifier", "logistic_regression")
        
        # Model 2: High Precision Model
        print("2. 🎯 Precision Specialist (Low False Positives)...")
        model2 = LogisticRegression(
            random_state=42,
            solver='liblinear',
            C=0.1,
            class_weight={0: 1, 1: 5},  # Strong bias towards catching fraud
            penalty='l1'
        )
        model2.fit(self.X_train, self.y_train)
        self.models["precision_specialist"] = model2
        self._initialize_model_info("precision_specialist", "Minimizes false positives", "logistic_regression_l1")
        
        # Model 3: High Recall Model
        print("3. 🎯 Recall Specialist (High Fraud Detection)...")
        model3 = RandomForestClassifier(
            n_estimators=150,
            random_state=42,
            max_depth=12,
            class_weight='balanced',
            min_samples_split=5,
            n_jobs=-1
        )
        model3.fit(self.X_train, self.y_train)
        self.models["recall_specialist"] = model3
        self._initialize_model_info("recall_specialist", "Maximizes fraud detection", "random_forest")
        
        # Model 4: Complex Pattern Detector
        print("4. 🎯 Complex Pattern Specialist (Gradient Boosting)...")
        model4 = GradientBoostingClassifier(
            n_estimators=200,
            random_state=42,
            learning_rate=0.05,
            max_depth=7,
            min_samples_split=10,
            subsample=0.8
        )
        model4.fit(self.X_train, self.y_train)
        self.models["complex_specialist"] = model4
        self._initialize_model_info("complex_specialist", "Detects complex patterns", "gradient_boosting")
        
        # Model 5: Drift Adaptive Model
        print("5. 🎯 Drift Adaptive Model (Neural Network)...")
        # Create robust training data
        X_augmented = np.vstack([
            self.X_train,
            self.X_train + np.random.normal(0, 0.15, self.X_train.shape),
            self.X_train * np.random.uniform(0.8, 1.2, self.X_train.shape)
        ])
        y_augmented = np.hstack([
            self.y_train,
            self.y_train,
            np.random.choice([0, 1], size=len(self.y_train), p=[0.7, 0.3])  # Add noise
        ])
        
        model5 = MLPClassifier(
            hidden_layer_sizes=(100, 50, 25),
            random_state=42,
            max_iter=1000,
            early_stopping=True,
            learning_rate='adaptive',
            alpha=0.001
        )
        model5.fit(X_augmented, y_augmented)
        self.models["drift_recovery"] = model5  # Fixed: Changed from drift_adaptive to drift_recovery
        self._initialize_model_info("drift_recovery", "Adapts to concept drift", "neural_network")
        
        # Save all models
        for name, model in self.models.items():
            joblib.dump(model, f'{name}_model.pkl')
            print(f"   💾 Saved: {name}_model.pkl")
        
        print(f"\n✅ Successfully trained and saved {len(self.models)} professional models")
    
    def _initialize_model_info(self, name, description, model_type):
        """Initialize model information structure"""
        self.model_info[name] = {
            "name": name,
            "description": description,
            "type": model_type,
            "created_at": datetime.now().isoformat(),  # Store as string
            "version": "1.0",
            "parameters": self._get_model_params(name),
            "performance": {
                "usage_count": 0,
                "total_predictions": 0,
                "correct_predictions": 0,
                "accuracy": 0.0,
                "avg_confidence": 0.0
            }
        }
    
    def _get_model_params(self, model_name):
        """Get model parameters for API"""
        if model_name not in self.models:
            return {}
        
        model = self.models[model_name]
        params = model.get_params() if hasattr(model, 'get_params') else {}
        
        # Clean up parameters for JSON serialization
        clean_params = {}
        for key, value in params.items():
            if isinstance(value, (int, float, str, bool, list, dict)):
                clean_params[key] = value
            else:
                clean_params[key] = str(value)
        
        return clean_params
    
    def get_best_model_for_situation(self, performance_metrics):
        """
        Intelligent model selection based on current performance
        """
        accuracy = performance_metrics.get('accuracy', 0)
        precision = performance_metrics.get('precision', 0)
        recall = performance_metrics.get('recall', 0)
        f1 = performance_metrics.get('f1_score', 0)
        
        # Decision logic
        if accuracy < 0.65:
            # Severe performance drop
            if precision < 0.4 and recall < 0.4:
                return "drift_recovery"  # Everything is broken
            elif precision < 0.5:
                return "recall_specialist"  # Need to catch more fraud
            elif recall < 0.5:
                return "precision_specialist"  # Too many false alarms
            else:
                return "complex_specialist"  # Complex patterns
        elif f1 < 0.6:
            # F1 score issues
            if precision < recall:
                return "precision_specialist"
            else:
                return "recall_specialist"
        else:
            # Normal operation - use weighted selection
            models = list(self.models.keys())
            weights = []
            
            for model in models:
                perf = self.model_info[model]['performance']['accuracy']
                usage = self.model_info[model]['performance']['usage_count']
                # Favor models with good performance and less usage
                weight = max(0.1, perf) * (1 / (usage + 1))
                weights.append(weight)
            
            if sum(weights) > 0:
                weights = np.array(weights) / sum(weights)
                return np.random.choice(models, p=weights)
            else:
                return "primary"
    
    def switch_model(self, new_model_name, reason="", performance_metrics=None):
        """
        Professional model switching with history tracking
        """
        if new_model_name not in self.models:
            print(f"❌ Model '{new_model_name}' not found!")
            return False
        
        old_model = self.current_model
        self.current_model = new_model_name
        
        # Update performance tracking
        self.model_info[new_model_name]['performance']['usage_count'] += 1
        
        # Record switch history
        switch_record = {
            "timestamp": datetime.now().isoformat(),  # Store as string
            "from": old_model,
            "to": new_model_name,
            "reason": reason,
            "performance_metrics": performance_metrics,
            "current_models": list(self.models.keys())
        }
        self.switch_history.append(switch_record)
        
        print(f"🔄 MODEL SWITCH: {old_model} → {new_model_name}")
        print(f"   Reason: {reason}")
        if performance_metrics:
            print(f"   Metrics: Accuracy={performance_metrics.get('accuracy', 0):.3f}, "
                  f"F1={performance_metrics.get('f1_score', 0):.3f}")
        
        return True
    
    def predict(self, features, return_details=True):
        """
        Professional prediction with detailed output
        """
        if self.current_model not in self.models:
            print(f"❌ Current model '{self.current_model}' not available!")
            return None
        
        model = self.models[self.current_model]
        
        try:
            # Ensure features is 2D array
            if len(features.shape) == 1:
                features = features.reshape(1, -1)
            
            # Get prediction
            prediction = model.predict(features)[0]
            
            # Get confidence/probability
            confidence = 0.5
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(features)[0]
                confidence = max(proba)
                probabilities = {
                    "genuine": float(proba[0]),
                    "fraud": float(proba[1])
                }
            elif hasattr(model, 'decision_function'):
                decision = model.decision_function(features)[0]
                confidence = 1 / (1 + np.exp(-np.abs(decision)))
                probabilities = {
                    "genuine": float(1 - confidence),
                    "fraud": float(confidence)
                }
            else:
                probabilities = {
                    "genuine": 0.5,
                    "fraud": 0.5
                }
            
            # Update performance tracking
            self._update_prediction_stats(prediction == 1, confidence)
            
            if return_details:
                return {
                    "prediction": int(prediction),
                    "label": "fraud" if prediction == 1 else "genuine",
                    "confidence": float(confidence),
                    "probabilities": probabilities,
                    "model_used": self.current_model,
                    "model_info": self.model_info[self.current_model]['description'],
                    "timestamp": datetime.now().isoformat()  # Store as string
                }
            else:
                return prediction
                
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            # Fallback mechanism
            if self.current_model != "primary":
                print("   Attempting fallback to primary model...")
                self.switch_model("primary", f"Fallback due to error: {str(e)}")
                return self.predict(features, return_details)
            else:
                # Ultimate fallback
                return {
                    "prediction": 0,
                    "label": "genuine",
                    "confidence": 0.5,
                    "probabilities": {"genuine": 0.5, "fraud": 0.5},
                    "model_used": "fallback",
                    "model_info": "Fallback system",
                    "timestamp": datetime.now().isoformat(),  # Store as string
                    "error": str(e)
                }
    
    def _update_prediction_stats(self, correct, confidence):
        """Update prediction statistics"""
        model_name = self.current_model
        perf = self.model_info[model_name]['performance']
        
        perf['total_predictions'] += 1
        if correct:
            perf['correct_predictions'] += 1
        
        perf['accuracy'] = perf['correct_predictions'] / perf['total_predictions'] if perf['total_predictions'] > 0 else 0
        perf['avg_confidence'] = (perf['avg_confidence'] * (perf['total_predictions'] - 1) + confidence) / perf['total_predictions']
    
    def get_system_status(self):
        """Get complete system status for API"""
        return {
            "status": "operational",
            "current_model": self.current_model,
            "available_models": list(self.models.keys()),
            "total_models": len(self.models),
            "switch_count": len(self.switch_history),
            "model_performance": {
                name: info['performance'] for name, info in self.model_info.items()
            },
            "last_switch": self.switch_history[-1] if self.switch_history else None,
            "timestamp": datetime.now().isoformat()  # Store as string
        }
    
    def export_model_info(self, filename="model_inventory.json"):
        """Export complete model information"""
        inventory = {
            "export_timestamp": datetime.now().isoformat(),  # Store as string
            "models": self.model_info,
            "switch_history": self.switch_history[-100:],  # Last 100 switches
            "system_status": self.get_system_status()
        }
        
        with open(filename, 'w') as f:
            json.dump(inventory, f, indent=2)
        
        print(f"📁 Model inventory exported to {filename}")
        return inventory
    
    def get_api_endpoints(self):
        """Get available API endpoints for frontend"""
        return [
            {
                "endpoint": "/api/models/status",
                "method": "GET",
                "description": "Get system status and current model"
            },
            {
                "endpoint": "/api/models/predict",
                "method": "POST",
                "description": "Make fraud prediction",
                "parameters": ["features"]
            },
            {
                "endpoint": "/api/models/history",
                "method": "GET",
                "description": "Get model switch history"
            },
            {
                "endpoint": "/api/models/list",
                "method": "GET",
                "description": "List all available models"
            }
        ]
    
    def batch_predict(self, features_list):
        """Batch prediction for multiple samples"""
        results = []
        for features in features_list:
            result = self.predict(features, return_details=True)
            results.append(result)
        return results