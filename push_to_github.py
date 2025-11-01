"""
Script to push project to GitHub repository

Run this script to automatically push all files to your GitHub repo.
"""

import os
import subprocess
import sys

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} - Success!")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - Failed!")
        print(f"Error: {e.stderr}")
        return False

def main():
    """Main function to push to GitHub."""
    print("=" * 80)
    print("🚀 PUSHING PROJECT TO GITHUB")
    print("=" * 80)
    
    repo_url = "https://github.com/dimpalmang20/Deepfake_Tool.git"
    
    # Check if git is installed
    if not run_command("git --version", "Checking Git installation"):
        print("\n❌ Git is not installed. Please install Git first:")
        print("   Download from: https://git-scm.com/downloads")
        return False
    
    # Initialize git if not already initialized
    if not os.path.exists(".git"):
        print("\n📁 Initializing Git repository...")
        run_command("git init", "Initializing Git")
    
    # Add all files
    print("\n📦 Adding all files...")
    run_command("git add .", "Adding files to Git")
    
    # Commit
    print("\n💾 Committing changes...")
    commit_message = "Add complete DeepFake Detection System - Production ready"
    run_command(f'git commit -m "{commit_message}"', "Committing changes")
    
    # Check if remote exists
    try:
        result = subprocess.run("git remote -v", shell=True, capture_output=True, text=True)
        if repo_url not in result.stdout:
            print(f"\n🔗 Adding remote repository...")
            run_command(f'git remote add origin {repo_url}', "Adding remote")
            run_command("git branch -M main", "Setting branch to main")
    except:
        pass
    
    # Push to GitHub
    print("\n🚀 Pushing to GitHub...")
    print(f"Repository: {repo_url}")
    print("\n⚠️  You may be asked to enter your GitHub credentials.")
    print("   If prompted:")
    print("   - Username: dimpalmang20")
    print("   - Password: Use a Personal Access Token (not your password)")
    print("   - Get token from: https://github.com/settings/tokens")
    
    success = run_command("git push -u origin main", "Pushing to GitHub")
    
    if success:
        print("\n" + "=" * 80)
        print("✅ SUCCESS! Project pushed to GitHub!")
        print("=" * 80)
        print(f"📁 Repository: {repo_url}")
        print("\n🎯 Next Step: Deploy to Railway")
        print("   1. Go to: https://railway.app")
        print("   2. Sign up/login")
        print("   3. New Project → Deploy from GitHub repo")
        print("   4. Select: dimpalmang20/Deepfake_Tool")
        print("   5. Wait 2-3 minutes")
        print("   6. Get your deployment link!")
        return True
    else:
        print("\n" + "=" * 80)
        print("❌ Push failed. Please check:")
        print("=" * 80)
        print("1. Git credentials are set up")
        print("2. Repository exists at: https://github.com/dimpalmang20/Deepfake_Tool")
        print("3. You have write access to the repository")
        print("4. Use Personal Access Token if password doesn't work")
        return False

if __name__ == "__main__":
    main()

