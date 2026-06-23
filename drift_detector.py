"""
Professional Drift Detection Module
Exports detection history for frontend visualization
"""

import numpy as np
import json
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from scipy import stats
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class AdvancedDriftDetector:
    def __init__(self, window_size=100, accuracy_threshold=0.75, confidence_level=0.95):
        """
        Professional drift detector with export capabilities
        """
        self.window_size = window_size
        self.accuracy_threshold = accuracy_threshold
        self.confidence_level = confidence_level
        
        # Prediction history
        self.predictions = []
        self.labels = []
        self.confidences = []
        
        # Statistical tracking with timestamps
        self.accuracy_history = []
        self.f1_history = []
        self.drift_alerts = []
        self.metrics_history = []
        
        # Performance statistics
        self.total_samples = 0
        self.total_drifts = 0
        
        print(f"🚀 Professional Drift Detector Initialized")
        print(f"   Window Size: {window_size}, Threshold: {accuracy_threshold}")

    def add_prediction(self, y_true, y_pred, confidence=None, timestamp=None):
        """
        Add prediction with timestamp for tracking
        """
        self.labels.append(y_true)
        self.predictions.append(y_pred)
        self.confidences.append(confidence if confidence is not None else 1.0)
        self.total_samples += 1
        
        # Maintain sliding window
        if len(self.labels) > self.window_size:
            self.labels.pop(0)
            self.predictions.pop(0)
            self.confidences.pop(0)
        
        # Record timestamp
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        return timestamp

    def check_drift(self, sample_index=None):
        """
        Comprehensive drift detection with detailed reporting
        """
        if len(self.labels) < self.window_size:
            return {
                "drift_detected": False, 
                "reason": "Insufficient data", 
                "confidence": 0.0,
                "status": "monitoring"
            }
        
        # Calculate comprehensive metrics
        accuracy = accuracy_score(self.labels, self.predictions)
        f1 = f1_score(self.labels, self.predictions, zero_division=0)
        precision = precision_score(self.labels, self.predictions, zero_division=0)
        recall = recall_score(self.labels, self.predictions, zero_division=0)
        
        # Store history
        self.accuracy_history.append(accuracy)
        self.f1_history.append(f1)
        
        metric_record = {
            "timestamp": datetime.now().isoformat(),  # Store as string
            "accuracy": float(accuracy),
            "f1_score": float(f1),
            "precision": float(precision),
            "recall": float(recall),
            "window_size": len(self.labels)
        }
        self.metrics_history.append(metric_record)
        
        # Multiple detection strategies
        accuracy_alert = accuracy < self.accuracy_threshold
        f1_alert = f1 < 0.5
        precision_alert = precision < 0.4
        recall_alert = recall < 0.3
        
        # Statistical test
        statistical_alert = False
        stat_confidence = 0.0
        
        if len(self.accuracy_history) > 50:
            recent_window = 20
            recent_acc = self.accuracy_history[-recent_window:]
            older_acc = self.accuracy_history[-2*recent_window:-recent_window]
            
            if len(recent_acc) > 10 and len(older_acc) > 10:
                t_stat, p_value = stats.ttest_ind(recent_acc, older_acc, equal_var=False)
                statistical_alert = p_value < (1 - self.confidence_level)
                stat_confidence = 1 - p_value
        
        # Determine drift
        drift_detected = accuracy_alert or f1_alert or statistical_alert
        
        if drift_detected:
            self.total_drifts += 1
            
            if accuracy_alert:
                reason = f"Accuracy dropped to {accuracy:.3f} (threshold: {self.accuracy_threshold})"
                severity = "high"
            elif f1_alert:
                reason = f"F1 score dropped to {f1:.3f}"
                severity = "medium"
            elif statistical_alert:
                reason = "Statistical distribution change detected"
                severity = "low"
            else:
                reason = "Performance degradation detected"
                severity = "medium"
            
            alert_record = {
                "alert_id": len(self.drift_alerts) + 1,
                "timestamp": datetime.now().isoformat(),  # Store as string
                "sample_index": sample_index if sample_index else len(self.labels),
                "accuracy": float(accuracy),
                "f1_score": float(f1),
                "precision": float(precision),
                "recall": float(recall),
                "reason": reason,
                "severity": severity,
                "confidence": float(max(stat_confidence, self.confidence_level)),
                "window_samples": len(self.labels)
            }
            self.drift_alerts.append(alert_record)
            
            print(f"⚠️ DRIFT ALERT #{len(self.drift_alerts)}: {reason}")
            print(f"   Accuracy: {accuracy:.3f}, F1: {f1:.3f}, Precision: {precision:.3f}")
        
        return {
            "drift_detected": drift_detected,
            "accuracy": float(accuracy),
            "f1_score": float(f1),
            "precision": float(precision),
            "recall": float(recall),
            "reason": "System stable" if not drift_detected else reason,
            "severity": "none" if not drift_detected else severity,
            "confidence": float(stat_confidence),
            "window_size": len(self.labels),
            "status": "stable" if not drift_detected else "alert"
        }

    def get_system_status(self):
        """Get current system status summary"""
        if len(self.accuracy_history) == 0:
            return {
                "status": "initializing",
                "message": "Collecting initial data",
                "samples_processed": self.total_samples
            }
        
        current_acc = self.accuracy_history[-1] if self.accuracy_history else 0
        status = "healthy" if current_acc >= self.accuracy_threshold else "degraded"
        
        return {
            "status": status,
            "current_accuracy": float(current_acc),
            "total_drifts_detected": self.total_drifts,
            "samples_processed": self.total_samples,
            "window_size": len(self.labels),
            "alerts_today": len([a for a in self.drift_alerts if datetime.fromisoformat(a['timestamp']).date() == datetime.now().date()])
        }

    def export_history(self, filename="drift_analysis.json"):
        """Export complete detection history to JSON file"""
        history = {
            "detector_config": {
                "window_size": self.window_size,
                "accuracy_threshold": self.accuracy_threshold,
                "confidence_level": self.confidence_level
            },
            "performance_summary": {
                "total_samples": self.total_samples,
                "total_drifts": self.total_drifts,
                "average_accuracy": float(np.mean(self.accuracy_history)) if self.accuracy_history else 0,
                "average_f1": float(np.mean(self.f1_history)) if self.f1_history else 0
            },
            "metrics_history": self.metrics_history[-1000:],  # Last 1000 records
            "drift_alerts": self.drift_alerts,
            "export_timestamp": datetime.now().isoformat()  # Store as string
        }
        
        with open(filename, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"📊 History exported to {filename}")
        return history

    def get_realtime_metrics(self, last_n=50):
        """Get recent metrics for realtime monitoring"""
        if len(self.metrics_history) == 0:
            return {"metrics": [], "summary": {}}
        
        recent = self.metrics_history[-last_n:]
        
        summary = {
            "avg_accuracy": float(np.mean([m['accuracy'] for m in recent])),
            "avg_f1": float(np.mean([m['f1_score'] for m in recent])),
            "trend": "improving" if len(self.accuracy_history) > 1 and self.accuracy_history[-1] > self.accuracy_history[-2] else "declining",
            "last_update": recent[-1]['timestamp'] if recent else None
        }
        
        return {
            "metrics": recent,
            "summary": summary
        }

    def get_performance_history(self):
        """Get complete performance history"""
        return {
            "accuracy_history": [float(x) for x in self.accuracy_history],
            "f1_history": [float(x) for x in self.f1_history],
            "drift_alerts": self.drift_alerts,
            "total_samples": self.total_samples,
            "config": {
                "window_size": self.window_size,
                "accuracy_threshold": self.accuracy_threshold
            }
        }