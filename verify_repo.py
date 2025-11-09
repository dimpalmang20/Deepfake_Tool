"""
Script to verify the GitHub repository contains all necessary files
"""

import os
from pathlib import Path

def verify_repository():
    """Verify all critical files are present."""
    print("=" * 80)
    print("🔍 VERIFYING GITHUB REPOSITORY")
    print("=" * 80)
    print(f"\n📁 Checking repository: https://github.com/dimpalmang20/Deepfake_Tool")
    print("-" * 80)
    
    # Critical files that must exist
    critical_files = [
        "main.py",                    # Main application
        "web_interface.html",        # Web UI
        "requirements.txt",           # Dependencies
        "requirements_clean.txt",    # Clean dependencies
        "Procfile",                   # Railway/Heroku
        "Dockerfile",                 # Docker deployment
        "README.md",                  # Documentation
        "DEPLOYMENT_GUIDE.md",        # Deployment guide
        "data/dataset.csv",           # Dataset
    ]
    
    # Core Python modules
    core_modules = [
        "data_loader.py",
        "train.py",
        "inference.py",
        "explainability.py",
        "temporal_model.py",
        "retrain_loop.py",
    ]
    
    # Utility files
    utils_files = [
        "utils/face_detection.py",
        "utils/frequency_analysis.py",
        "utils/evaluation_metrics.py",
    ]
    
    print("\n✅ Checking Critical Files:")
    print("-" * 40)
    all_critical = True
    for file in critical_files:
        if os.path.exists(file):
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file} - MISSING!")
            all_critical = False
    
    print("\n✅ Checking Core Modules:")
    print("-" * 40)
    for module in core_modules:
        if os.path.exists(module):
            print(f"   ✅ {module}")
        else:
            print(f"   ❌ {module} - MISSING!")
    
    print("\n✅ Checking Utils:")
    print("-" * 40)
    for util in utils_files:
        if os.path.exists(util):
            print(f"   ✅ {util}")
        else:
            print(f"   ❌ {util} - MISSING!")
    
    # Check dataset
    print("\n✅ Checking Dataset:")
    print("-" * 40)
    dataset_exists = os.path.exists("data/dataset.csv")
    real_images = len(list(Path("data/images/real").glob("*.jpg"))) if Path("data/images/real").exists() else 0
    fake_images = len(list(Path("data/images/fake").glob("*.jpg"))) if Path("data/images/fake").exists() else 0
    real_videos = len(list(Path("data/videos/real").glob("*.mp4"))) if Path("data/videos/real").exists() else 0
    fake_videos = len(list(Path("data/videos/fake").glob("*.mp4"))) if Path("data/videos/fake").exists() else 0
    
    if dataset_exists:
        print(f"   ✅ dataset.csv exists")
    else:
        print(f"   ❌ dataset.csv - MISSING!")
    
    print(f"   ✅ Real Images: {real_images}")
    print(f"   ✅ Fake Images: {fake_images}")
    print(f"   ✅ Real Videos: {real_videos}")
    print(f"   ✅ Fake Videos: {fake_videos}")
    
    total_samples = real_images + fake_images + real_videos + fake_videos
    print(f"   📊 Total Samples: {total_samples}")
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 80)
    
    if all_critical and dataset_exists and total_samples > 0:
        print("✅ REPOSITORY VERIFICATION: SUCCESS!")
        print("\n✅ All critical files present")
        print("✅ Dataset with samples ready")
        print("✅ Ready for deployment!")
        print("\n🎯 Next Step: Deploy to Railway")
        print("   1. Go to: https://railway.app")
        print("   2. Login with GitHub")
        print("   3. New Project → Deploy from GitHub repo")
        print("   4. Select: dimpalmang20/Deepfake_Tool")
        print("   5. Wait 2-3 minutes")
        print("   6. Get your deployment link!")
        return True
    else:
        print("⚠️  REPOSITORY VERIFICATION: Some files may be missing")
        print("   Check the list above and ensure all files are pushed to GitHub")
        return False

if __name__ == "__main__":
    verify_repository()





