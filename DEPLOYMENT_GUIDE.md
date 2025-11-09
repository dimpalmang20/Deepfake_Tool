# 🚀 DeepFake Detection System - Deployment Guide

## 🌐 Quick Deploy Options

### Option 1: Railway.app (Recommended - FREE & Easy)

**Step-by-Step:**

1. **Sign up at Railway.app** (https://railway.app)
   - Click "Login with GitHub"
   - Authorize Railway

2. **Create New Project**
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your repository

3. **Deploy**
   - Railway will automatically detect Python
   - It will use `main.py` and `requirements_clean.txt`
   - Wait 2-3 minutes for deployment

4. **Get Your Link**
   - Click on your service
   - Go to "Settings" → "Networking"
   - Generate domain (or use custom domain)
   - **Your link will be: `https://your-app-name.up.railway.app`**

**Railway automatically:**
- ✅ Installs dependencies
- ✅ Runs `python main.py`
- ✅ Provides HTTPS URL
- ✅ Handles restarts

---

### Option 2: Render.com (FREE Tier)

**Step-by-Step:**

1. **Sign up at Render.com** (https://render.com)
   - Connect your GitHub account

2. **Create New Web Service**
   - Click "New" → "Web Service"
   - Connect your repository
   - Render will auto-detect settings

3. **Configure**
   - **Name**: deepfake-detector
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements_clean.txt`
   - **Start Command**: `python main.py`

4. **Deploy**
   - Click "Create Web Service"
   - Wait 5-7 minutes
   - **Your link will be: `https://deepfake-detector.onrender.com`**

---

### Option 3: Fly.io (FREE Tier)

**Step-by-Step:**

1. **Install Fly CLI**
   ```bash
   # Windows
   powershell -Command "iwr https://fly.io/install.ps1 -useb | iex"
   ```

2. **Login to Fly**
   ```bash
   fly auth signup
   ```

3. **Deploy**
   ```bash
   fly launch
   # Follow prompts, select Python
   # Your app will be deployed!
   ```

4. **Get URL**
   - Your app will be at: `https://your-app-name.fly.dev`

---

### Option 4: Local Deployment (For Testing)

**Step-by-Step:**

1. **Install Dependencies**
   ```bash
   pip install -r requirements_clean.txt
   ```

2. **Run Application**
   ```bash
   python main.py
   ```

3. **Open Browser**
   - Go to: `http://localhost:8000`
   - Or visit: `http://localhost:8000/docs` for API documentation

---

## 📋 Required Files for Deployment

Make sure these files exist:
- ✅ `main.py` - Main application
- ✅ `requirements_clean.txt` - Dependencies
- ✅ `web_interface.html` - Web UI (optional)
- ✅ `Procfile` - For Heroku/Railway (already created)

---

## 🔧 Environment Variables

No environment variables required! The app works out of the box.

Optional:
- `PORT` - Server port (default: 8000)

---

## ✅ After Deployment

Your deployed app will have:

1. **Web Interface**: `https://your-url.com`
   - Upload images/videos
   - See detection results
   - View explanations

2. **API Documentation**: `https://your-url.com/docs`
   - Interactive API testing
   - All endpoints documented

3. **Health Check**: `https://your-url.com/health`
   - Check if system is running

---

## 🎯 Testing Your Deployment

1. **Open your deployment URL**
2. **Upload a test image**
3. You should see:
   - ✅ Prediction (Real/Fake)
   - ✅ Confidence score
   - ✅ Explanation
   - ✅ Frequency analysis

---

## 📞 Support

If deployment fails:
1. Check logs in your platform's dashboard
2. Verify `requirements_clean.txt` is correct
3. Ensure `main.py` exists
4. Check Python version (3.10+ required)

---

## 🎉 Success!

Once deployed, share your link:
**`https://your-deployment-url.com`**

Your DeepFake Detection System is live and ready to use! 🚀






