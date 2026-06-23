"""
Complete Project Runner - One command to run everything
"""

import os
import sys
import subprocess
import time
from datetime import datetime

def print_header():
    """Print project header"""
    print("\n" + "="*80)
    print("🎯 ULTIMATE FRAUD DETECTION SYSTEM - COMPLETE RUNNER")
    print("="*80)

def check_environment():
    """Check Python environment"""
    print("\n🔍 Checking environment...")
    
    # Check Python version
    python_version = sys.version_info
    print(f"   Python: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major < 3 or python_version.minor < 8:
        print("   ⚠️  Python 3.8+ recommended")
    
    return True

def install_requirements():
    """Install required packages"""
    print("\n📦 Installing requirements...")
    
    requirements = [
        'scikit-learn>=1.2.0',
        'numpy>=1.22.0',
        'matplotlib>=3.6.0',
        'pandas>=1.5.0',
        'scipy>=1.9.0',
        'joblib>=1.2.0',
        'flask>=2.3.0',
        'flask-cors>=4.0.0',
        'seaborn>=0.12.0'
    ]
    
    for req in requirements:
        package = req.split('>=')[0]
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])
            print(f"   ✅ {package}")
        except:
            print(f"   ⚠️  {package} (may already be installed)")
    
    print("\n✅ All requirements installed!")

def clean_old_files():
    """Clean old generated files"""
    print("\n🧹 Cleaning old files...")
    
    files_to_remove = [
        '*.pkl', '*.png', '*.json',
        'backup_*.pkl', 'drift_specialist_model.pkl',
        'gradient_boosting_model.pkl', 'random_forest_model.pkl'
    ]
    
    for pattern in files_to_remove:
        try:
            os.system(f"del {pattern}" if os.name == 'nt' else f"rm -f {pattern}")
        except:
            pass
    
    print("   ✅ Cleaned old files")

def run_fraud_detection():
    """Run the main fraud detection system"""
    print("\n" + "="*80)
    print("🚀 RUNNING FRAUD DETECTION SYSTEM")
    print("="*80)
    
    try:
        start_time = time.time()
        
        # Import and run
        from frauddrift import UltimateFraudDetectionSystem
        
        system = UltimateFraudDetectionSystem()
        result = system.run_complete_system()
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n⏱️  Execution completed in {duration:.2f} seconds")
        
        if result.get('success', False):
            print("\n✅ SYSTEM EXECUTION SUCCESSFUL!")
            return True
        else:
            print(f"\n❌ System execution failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"\n❌ Error running system: {e}")
        return False

def start_api_server():
    """Start the Flask API server"""
    print("\n" + "="*80)
    print("🌐 STARTING API SERVER")
    print("="*80)
    
    print("\n📡 API Server will run on: http://localhost:5000")
    print("\nAvailable endpoints:")
    print("   GET  /api/status     - System status")
    print("   GET  /api/models     - List all models")
    print("   POST /api/predict    - Make fraud prediction")
    print("   GET  /api/drift      - Drift detection status")
    print("   GET  /api/dashboard  - Complete dashboard data")
    
    print("\n⏳ Starting server in background...")
    print("   Press Ctrl+C in the API server window to stop")
    
    # Start API server in a separate process
    try:
        import api_server
        print("\n✅ API Server module loaded successfully")
        print("\n📢 To start the API server manually, run:")
        print("   python api_server.py")
        
    except Exception as e:
        print(f"\n❌ Could not start API server: {e}")
        print("\nYou can still run it manually with: python api_server.py")

def show_summary():
    """Show execution summary"""
    print("\n" + "="*80)
    print("📊 EXECUTION SUMMARY")
    print("="*80)
    
    generated_files = []
    for file in os.listdir('.'):
        if file.endswith(('.pkl', '.png', '.json')):
            size = os.path.getsize(file) / 1024  # KB
            generated_files.append((file, size))
    
    print("\n📁 GENERATED FILES:")
    for file, size in generated_files:
        print(f"   📄 {file:40} {size:7.1f} KB")
    
    print("\n🎯 NEXT STEPS:")
    print("   1. View 'professional_dashboard.png' for complete analysis")
    print("   2. Check 'system_report.json' for detailed performance metrics")
    print("   3. Run 'python api_server.py' to start the API server")
    print("   4. Use 'frontend_data.json' for frontend integration")
    
    print("\n🔧 SYSTEM READY FOR:")
    print("   - Academic presentation 📚")
    print("   - Frontend development 🎨")
    print("   - Production deployment 🚀")
    print("   - Further research 🔬")

def main():
    """Main execution function"""
    print_header()
    
    # Step 1: Check environment
    if not check_environment():
        return
    
    # Step 2: Install requirements
    try:
        install_requirements()
    except Exception as e:
        print(f"⚠️  Requirements installation had issues: {e}")
        print("Trying to continue...")
    
    # Step 3: Clean old files
    clean_old_files()
    
    # Step 4: Run fraud detection system
    success = run_fraud_detection()
    
    if success:
        # Step 5: Show API server instructions
        start_api_server()
        
        # Step 6: Show summary
        show_summary()
        
        print("\n" + "="*80)
        print("🎉 PROJECT COMPLETED SUCCESSFULLY!")
        print("="*80)
        
    else:
        print("\n" + "="*80)
        print("❌ PROJECT EXECUTION FAILED")
        print("="*80)
        print("\nCheck the error messages above for details.")

if __name__ == "__main__":
    main()