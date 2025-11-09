# 🚀 Vercel Deployment Guide - Step by Step

## ⚠️ IMPORTANT: PyTorch Size Limitations

**PyTorch is very large (several GB)** and may exceed Vercel's serverless function size limits:
- **Hobby Plan**: 50MB limit ❌ (PyTorch won't fit)
- **Pro Plan**: 250MB limit ❌ (PyTorch won't fit)
- **Enterprise**: Custom limits (may work)

**Recommendation**: Use **Railway.app** or **Render.com** instead - they have no size limits and work better for ML applications.

However, if you still want to try Vercel (for Pro/Enterprise accounts), follow the steps below.

---

## 📋 Prerequisites

1. ✅ Code pushed to GitHub
2. ✅ Vercel account (Pro or Enterprise recommended)
3. ✅ GitHub repository access

---

## 🚀 Deployment Steps (5 Minutes)

### Step 1: Push Code to GitHub

If you haven't already, push your code:

```bash
cd DeepFake_Detector
git add .
git commit -m "Prepare for Vercel deployment"
git push origin main
```

---

### Step 2: Connect GitHub to Vercel

1. **Go to Vercel Dashboard**
   - Visit: https://vercel.com
   - Click **"Login"** or **"Sign Up"**
   - Sign in with your **GitHub account**

2. **Import Your Project**
   - Click **"Add New..."** → **"Project"**
   - Find your repository: `dimpalmang20/Deepfake_Tool`
   - Click **"Import"**

---

### Step 3: Configure Project Settings

Vercel should auto-detect the settings, but verify:

1. **Framework Preset**: Leave as **"Other"** (or "FastAPI" if available)
2. **Root Directory**: `DeepFake_Detector` (if your repo has subdirectories)
3. **Build Command**: Leave empty (not needed for Python)
4. **Output Directory**: Leave empty
5. **Install Command**: Leave empty (Vercel auto-installs from requirements.txt)

**Environment Variables**: None required for basic deployment

---

### Step 4: Deploy

1. Click **"Deploy"**
2. Wait 3-5 minutes for:
   - Dependencies installation
   - Build process
   - Deployment

---

### Step 5: Get Your Deployment Link

After deployment completes:

1. Vercel will show your deployment URL
2. Format: `https://your-project-name.vercel.app`
3. Click the URL to visit your deployed app

---

## 🔧 Troubleshooting

### Issue 1: "Package size too large" Error

**Problem**: PyTorch exceeds Vercel's size limits

**Solutions**:
1. **Upgrade to Vercel Pro/Enterprise** (if available)
2. **Use Railway or Render** instead (recommended)
3. **Use a lighter ML framework** (TensorFlow Lite, ONNX Runtime)
4. **Load models from external storage** (AWS S3, Cloud Storage)

### Issue 2: "Import Error" or "Module Not Found"

**Problem**: Python modules not found

**Solutions**:
1. Check `requirements.txt` has all dependencies
2. Verify `api/index.py` exists and is correct
3. Check Vercel build logs for specific errors
4. Ensure `vercel.json` is configured correctly

### Issue 3: "Function Timeout"

**Problem**: Requests take too long

**Solutions**:
1. Increase timeout in `vercel.json` (max 60s for Pro)
2. Optimize model inference
3. Use caching for repeated requests

### Issue 4: "File Not Found" (web_interface.html)

**Problem**: Static files not found

**Solutions**:
1. Check `web_interface.html` is in the project root
2. Verify file paths in `main.py`
3. The fallback HTML will be used if file not found

---

## 📁 Project Structure for Vercel

```
DeepFake_Detector/
├── api/
│   └── index.py          # Vercel serverless function handler
├── main.py               # FastAPI app
├── web_interface.html    # Web UI
├── requirements.txt      # Python dependencies
├── vercel.json           # Vercel configuration
└── ... (other files)
```

---

## ✅ Verification Checklist

After deployment, verify:

- [ ] Deployment completed successfully
- [ ] No build errors in Vercel logs
- [ ] Can access the deployment URL
- [ ] Home page loads (web interface or API docs)
- [ ] `/health` endpoint returns 200 OK
- [ ] `/docs` shows FastAPI documentation
- [ ] Can upload and detect images

---

## 🔗 Quick Links

- **Vercel Dashboard**: https://vercel.com/dashboard
- **Project Settings**: https://vercel.com/dashboard → Your Project → Settings
- **Deployment Logs**: https://vercel.com/dashboard → Your Project → Deployments → Click Deployment → Logs
- **API Documentation**: `https://your-project.vercel.app/docs`

---

## 🎯 Alternative: Railway Deployment (Recommended)

If Vercel doesn't work due to size limits, use **Railway**:

1. Go to https://railway.app
2. Login with GitHub
3. Click "New Project" → "Deploy from GitHub repo"
4. Select your repository
5. Railway auto-detects Python and deploys
6. Get your deployment link in 2-3 minutes

**Railway has no size limits and works perfectly for ML applications!**

---

## 📝 Notes

- Vercel uses Python 3.12 by default
- Dependencies are installed from `requirements.txt`
- The `api/index.py` file is the serverless function entry point
- Static files are served from the project root
- Maximum function timeout: 60 seconds (Pro plan)
- Maximum function size: 250MB (Pro plan)

---

## 🆘 Need Help?

If you encounter issues:

1. Check Vercel build logs for errors
2. Verify all files are pushed to GitHub
3. Check `requirements.txt` for correct dependencies
4. Try Railway or Render as alternatives
5. Check the troubleshooting section above

---

## 🎉 Success!

Once deployed, you'll have:
- ✅ Live deployment URL
- ✅ Working DeepFake Detection API
- ✅ Web interface (if file found)
- ✅ API documentation at `/docs`
- ✅ Health check at `/health`

**Share your deployment link and start detecting deepfakes!** 🚀

