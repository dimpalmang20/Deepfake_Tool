"""
Vercel Deployment Helper Script

This script helps prepare and deploy your DeepFake Detection System to Vercel.
It checks prerequisites, validates configuration, and provides deployment instructions.
"""

import os
import sys
import subprocess
from pathlib import Path

def print_banner():
    """Print deployment banner."""
    print("=" * 80)
    print("🚀 VERCEL DEPLOYMENT HELPER")
    print("=" * 80)
    print()

def check_files():
    """Check if all required files exist."""
    print("📁 Checking required files...")
    
    required_files = [
        "requirements.txt",
        "vercel.json",
        "api/index.py",
        "main.py",
        "web_interface.html"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
            print(f"   ❌ {file} - NOT FOUND")
        else:
            print(f"   ✅ {file} - Found")
    
    if missing_files:
        print(f"\n❌ Missing files: {', '.join(missing_files)}")
        return False
    
    print("\n✅ All required files present!")
    return True

def check_requirements():
    """Check requirements.txt for Vercel compatibility."""
    print("\n📦 Checking requirements.txt...")
    
    try:
        with open("requirements.txt", "r") as f:
            content = f.read()
        
        # Check for Python 3.12 compatibility
        if "torch>=2.2.0" in content or "torch==2.2.0" in content:
            print("   ✅ PyTorch version is Python 3.12 compatible")
        else:
            print("   ⚠️  PyTorch version may not be Python 3.12 compatible")
        
        # Check for opencv-python-headless
        if "opencv-python-headless" in content:
            print("   ✅ Using opencv-python-headless (better for serverless)")
        elif "opencv-python" in content:
            print("   ⚠️  Consider using opencv-python-headless for serverless")
        
        # Check for numpy compatibility
        if "numpy>=1.26.0" in content:
            print("   ✅ NumPy version is Python 3.12 compatible")
        
        print("\n✅ Requirements.txt looks good!")
        return True
        
    except Exception as e:
        print(f"   ❌ Error reading requirements.txt: {e}")
        return False

def check_vercel_config():
    """Check vercel.json configuration."""
    print("\n⚙️  Checking vercel.json...")
    
    try:
        import json
        with open("vercel.json", "r") as f:
            config = json.load(f)
        
        # Check for required fields
        if "builds" in config:
            print("   ✅ Builds configuration found")
        else:
            print("   ⚠️  No builds configuration found")
        
        if "routes" in config:
            print("   ✅ Routes configuration found")
        else:
            print("   ⚠️  No routes configuration found")
        
        if "functions" in config:
            print("   ✅ Functions configuration found")
            if "api/index.py" in config.get("functions", {}):
                func_config = config["functions"]["api/index.py"]
                if "maxDuration" in func_config:
                    print(f"   ✅ Max duration: {func_config['maxDuration']}s")
                if "memory" in func_config:
                    print(f"   ✅ Memory: {func_config['memory']}MB")
        else:
            print("   ⚠️  No functions configuration found")
        
        print("\n✅ vercel.json configuration looks good!")
        return True
        
    except Exception as e:
        print(f"   ❌ Error reading vercel.json: {e}")
        return False

def check_api_handler():
    """Check api/index.py handler."""
    print("\n🔧 Checking API handler...")
    
    try:
        with open("api/index.py", "r") as f:
            content = f.read()
        
        if "from main import app" in content:
            print("   ✅ Correctly imports app from main.py")
        else:
            print("   ⚠️  May not correctly import app")
        
        if "handler = app" in content:
            print("   ✅ Exports handler for Vercel")
        else:
            print("   ⚠️  May not export handler correctly")
        
        print("\n✅ API handler looks good!")
        return True
        
    except Exception as e:
        print(f"   ❌ Error reading api/index.py: {e}")
        return False

def check_git():
    """Check if git is initialized and has remote."""
    print("\n🔍 Checking Git configuration...")
    
    try:
        # Check if git is installed
        result = subprocess.run(["git", "--version"], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            print("   ❌ Git is not installed")
            return False
        print("   ✅ Git is installed")
        
        # Check if git repo is initialized
        if not os.path.exists(".git"):
            print("   ⚠️  Git repository not initialized")
            print("   💡 Run: git init")
            return False
        print("   ✅ Git repository initialized")
        
        # Check for remote
        result = subprocess.run(["git", "remote", "-v"], 
                              capture_output=True, text=True, timeout=5)
        if "origin" not in result.stdout:
            print("   ⚠️  No remote repository configured")
            print("   💡 Run: git remote add origin <your-repo-url>")
            return False
        print("   ✅ Remote repository configured")
        
        print("\n✅ Git configuration looks good!")
        return True
        
    except Exception as e:
        print(f"   ⚠️  Could not check Git: {e}")
        return False

def show_warnings():
    """Show important warnings about Vercel deployment."""
    print("\n" + "=" * 80)
    print("⚠️  IMPORTANT WARNINGS")
    print("=" * 80)
    print()
    print("1. 📦 PYTORCH SIZE LIMITATIONS:")
    print("   - PyTorch is very large (several GB)")
    print("   - Vercel Hobby: 50MB limit ❌ (PyTorch won't fit)")
    print("   - Vercel Pro: 250MB limit ❌ (PyTorch won't fit)")
    print("   - Vercel Enterprise: Custom limits (may work)")
    print()
    print("   💡 RECOMMENDATION: Use Railway.app or Render.com instead!")
    print("      They have no size limits and work better for ML applications.")
    print()
    print("2. 🚀 DEPLOYMENT STEPS:")
    print("   1. Push code to GitHub")
    print("   2. Go to https://vercel.com")
    print("   3. Login with GitHub")
    print("   4. Import your repository")
    print("   5. Deploy (may fail due to size limits)")
    print()
    print("3. 🔄 ALTERNATIVE: Railway Deployment (Recommended)")
    print("   - Go to https://railway.app")
    print("   - Login with GitHub")
    print("   - Deploy from GitHub repo")
    print("   - No size limits, works perfectly!")
    print()
    print("=" * 80)

def show_deployment_instructions():
    """Show step-by-step deployment instructions."""
    print("\n" + "=" * 80)
    print("📋 DEPLOYMENT INSTRUCTIONS")
    print("=" * 80)
    print()
    print("STEP 1: Push Code to GitHub")
    print("   git add .")
    print("   git commit -m 'Prepare for Vercel deployment'")
    print("   git push origin main")
    print()
    print("STEP 2: Deploy to Vercel")
    print("   1. Go to https://vercel.com")
    print("   2. Login with GitHub")
    print("   3. Click 'Add New...' → 'Project'")
    print("   4. Import your repository")
    print("   5. Click 'Deploy'")
    print("   6. Wait for deployment (3-5 minutes)")
    print("   7. Get your deployment URL")
    print()
    print("STEP 3: Verify Deployment")
    print("   - Check deployment URL")
    print("   - Visit /health endpoint")
    print("   - Visit /docs for API documentation")
    print()
    print("=" * 80)

def main():
    """Main deployment helper function."""
    print_banner()
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Run checks
    all_checks_passed = True
    
    if not check_files():
        all_checks_passed = False
    
    if not check_requirements():
        all_checks_passed = False
    
    if not check_vercel_config():
        all_checks_passed = False
    
    if not check_api_handler():
        all_checks_passed = False
    
    git_ok = check_git()
    
    # Show warnings
    show_warnings()
    
    # Show deployment instructions
    show_deployment_instructions()
    
    # Final summary
    print("\n" + "=" * 80)
    print("📊 DEPLOYMENT READINESS SUMMARY")
    print("=" * 80)
    print()
    
    if all_checks_passed:
        print("✅ All configuration checks passed!")
        print("✅ Your project is ready for deployment!")
    else:
        print("⚠️  Some checks failed. Please fix the issues above.")
    
    if git_ok:
        print("✅ Git is configured and ready to push")
    else:
        print("⚠️  Git needs to be configured")
    
    print()
    print("🚀 NEXT STEPS:")
    print("   1. Fix any issues shown above")
    print("   2. Push code to GitHub")
    print("   3. Deploy to Vercel (or use Railway as alternative)")
    print()
    print("💡 TIP: Railway.app is recommended for ML applications!")
    print("   - No size limits")
    print("   - Better Python support")
    print("   - Easier deployment")
    print()
    print("=" * 80)

if __name__ == "__main__":
    main()

