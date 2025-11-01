"""
Quick Deployment Script for DeepFake Detection System

This script creates a simple deployment without external dependencies.
"""

import os
import sys
import subprocess
import time
import webbrowser
from pathlib import Path

def print_banner():
    """Print deployment banner."""
    print("=" * 80)
    print("🚀 DEEPFAKE DETECTION SYSTEM - QUICK DEPLOY")
    print("=" * 80)
    print("🌐 Creating production-ready deployment")
    print("📱 Web interface with real-time detection")
    print("🔬 Advanced AI with theoretical explanations")
    print("=" * 80)

def check_system():
    """Check system requirements."""
    print("\n🔍 Checking system requirements...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major == 3 and python_version.minor >= 8:
        print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    else:
        print(f"❌ Python {python_version.major}.{python_version.minor}.{python_version.micro} (Need 3.8+)")
        return False
    
    # Check if required files exist
    required_files = [
        "app_fixed.py",
        "web_interface.html",
        "requirements_clean.txt"
    ]
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - Missing")
            return False
    
    return True

def install_dependencies():
    """Install required dependencies."""
    print("\n📦 Installing dependencies...")
    
    try:
        # Install from clean requirements
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements_clean.txt"
        ], check=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def start_application():
    """Start the application."""
    print("\n🚀 Starting DeepFake Detection System...")
    
    try:
        # Start the application
        process = subprocess.Popen([
            sys.executable, "app_fixed.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for startup
        print("⏳ Starting server... Please wait...")
        time.sleep(5)
        
        # Check if process is running
        if process.poll() is None:
            print("✅ Application started successfully!")
            print("📱 Web Interface: http://localhost:8000")
            print("📚 API Documentation: http://localhost:8000/docs")
            print("🔍 Health Check: http://localhost:8000/health")
            
            # Open browser
            print("🌐 Opening web interface in browser...")
            webbrowser.open("http://localhost:8000")
            
            return process
        else:
            print("❌ Application failed to start")
            return None
            
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        return None

def create_deployment_package():
    """Create deployment package."""
    print("\n📁 Creating deployment package...")
    
    # Create deployment directory
    deploy_dir = "deployment_package"
    os.makedirs(deploy_dir, exist_ok=True)
    
    # Copy essential files
    essential_files = [
        "app_fixed.py",
        "web_interface.html",
        "requirements_clean.txt",
        "Dockerfile",
        "docker-compose.yml"
    ]
    
    for file in essential_files:
        if os.path.exists(file):
            subprocess.run(["cp", file, deploy_dir], check=True)
            print(f"✅ Copied {file}")
    
    # Create startup script
    startup_script = """#!/bin/bash
echo "🚀 Starting DeepFake Detection System..."
echo "📦 Installing dependencies..."
pip install -r requirements_clean.txt
echo "🌐 Starting web server..."
python app_fixed.py
"""
    
    with open(f"{deploy_dir}/start.sh", "w") as f:
        f.write(startup_script)
    
    # Make executable
    os.chmod(f"{deploy_dir}/start.sh", 0o755)
    print("✅ Created startup script")
    
    # Create README
    readme_content = """# DeepFake Detection System - Deployment Package

## 🚀 Quick Start

1. Install Python 3.8+
2. Run: `python app_fixed.py`
3. Open: http://localhost:8000

## 📦 Docker Deployment

```bash
docker build -t deepfake-detector .
docker run -p 8000:8000 deepfake-detector
```

## 🌐 Features

- Real-time deepfake detection
- Theoretical explanations
- Frequency domain analysis
- Grad-CAM visualizations
- Production-ready API

## 📱 Usage

1. Upload an image or video
2. Click "Detect Deepfake"
3. View results with explanations
4. Explore the theoretical approach

Your DeepFake Detection System is ready! 🎉
"""
    
    with open(f"{deploy_dir}/README.md", "w") as f:
        f.write(readme_content)
    
    print("✅ Created README.md")
    print(f"📁 Deployment package created in: {deploy_dir}")

def show_deployment_options():
    """Show deployment options."""
    print("\n🌐 DEPLOYMENT OPTIONS:")
    print("-" * 50)
    
    print("1. 🏠 LOCAL DEPLOYMENT (Current)")
    print("   • Run: python app_fixed.py")
    print("   • Access: http://localhost:8000")
    print("   • Best for: Development and testing")
    
    print("\n2. 🐳 DOCKER DEPLOYMENT")
    print("   • Run: docker build -t deepfake-detector .")
    print("   • Run: docker run -p 8000:8000 deepfake-detector")
    print("   • Best for: Production deployment")
    
    print("\n3. ☁️ CLOUD DEPLOYMENT")
    print("   • Railway: railway.app (Free)")
    print("   • Heroku: heroku.com (Free tier)")
    print("   • Render: render.com (Free tier)")
    print("   • Best for: Public access")
    
    print("\n4. 📦 DEPLOYMENT PACKAGE")
    print("   • Created in: deployment_package/")
    print("   • Contains: All necessary files")
    print("   • Best for: Sharing and distribution")

def main():
    """Main deployment function."""
    print_banner()
    
    # Check system
    if not check_system():
        print("\n❌ System requirements not met. Please fix issues and try again.")
        return
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Failed to install dependencies. Please check requirements_clean.txt")
        return
    
    # Create deployment package
    create_deployment_package()
    
    # Show deployment options
    show_deployment_options()
    
    # Ask user what to do
    print("\n" + "=" * 80)
    print("🎯 WHAT WOULD YOU LIKE TO DO?")
    print("=" * 80)
    print("1. Start local server now")
    print("2. Show deployment instructions only")
    print("3. Create deployment package only")
    print("=" * 80)
    
    choice = input("Choose option (1-3): ").strip()
    
    if choice == "1":
        process = start_application()
        
        if process:
            print("\n🎉 DEPLOYMENT SUCCESSFUL!")
            print("=" * 50)
            print("✅ DeepFake Detection System is running")
            print("🌐 Web Interface: http://localhost:8000")
            print("📚 API Documentation: http://localhost:8000/docs")
            print("🔍 Health Check: http://localhost:8000/health")
            print("\n🎯 Features Available:")
            print("   • Real-time deepfake detection")
            print("   • Theoretical explanations")
            print("   • Frequency domain analysis")
            print("   • Grad-CAM visualizations")
            print("   • Production-ready API")
            print("\n📱 Upload images/videos to test the system!")
            print("\nPress Ctrl+C to stop the server")
            
            try:
                process.wait()
            except KeyboardInterrupt:
                print("\n🛑 Stopping server...")
                process.terminate()
                print("✅ Server stopped successfully!")
    
    elif choice == "2":
        print("\n📋 DEPLOYMENT INSTRUCTIONS:")
        print("-" * 50)
        print("1. 🏠 LOCAL: python app_fixed.py")
        print("2. 🐳 DOCKER: docker build -t deepfake-detector . && docker run -p 8000:8000 deepfake-detector")
        print("3. ☁️ CLOUD: Upload deployment_package/ to your preferred platform")
        print("\n🎯 Your system is ready for deployment!")
    
    elif choice == "3":
        print("\n📦 DEPLOYMENT PACKAGE CREATED!")
        print("-" * 50)
        print("📁 Location: deployment_package/")
        print("📋 Contains: All necessary files for deployment")
        print("🚀 Ready for: Local, Docker, or Cloud deployment")
        print("\n🎯 Share this package with your team!")
    
    else:
        print("❌ Invalid choice")
        return
    
    print("\n🎉 DeepFake Detection System is ready for production! 🚀")

if __name__ == "__main__":
    main()





