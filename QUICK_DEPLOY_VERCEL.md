# 🚀 Quick Deploy to Vercel - 3 Steps

## ⚠️ IMPORTANT: PyTorch Size Warning

**PyTorch is too large for Vercel!** It will likely fail due to size limits.
**Recommendation: Use Railway.app instead** (no size limits, easier deployment)

---

## ✅ Step 1: Push Code to GitHub (2 minutes)

```bash
cd DeepFake_Detector

# Check deployment readiness
python deploy_vercel.py

# Push to GitHub
git add .
git commit -m "Prepare for Vercel deployment - Python 3.12 compatible"
git push origin main
```

**If you need to set up Git:**
```bash
git init
git remote add origin https://github.com/dimpalmang20/Deepfake_Tool.git
git branch -M main
git push -u origin main
```

---

## ✅ Step 2: Deploy on Vercel (3 minutes)

1. **Go to Vercel**
   - Visit: https://vercel.com
   - Click **"Login"** or **"Sign Up"**
   - **Login with GitHub** (use your GitHub account)

2. **Import Project**
   - Click **"Add New..."** → **"Project"**
   - Find your repository: `dimpalmang20/Deepfake_Tool`
   - Click **"Import"**

3. **Configure (Usually Auto-Detected)**
   - Framework Preset: **"Other"** (or leave default)
   - Root Directory: `DeepFake_Detector` (if your repo has subdirectories)
   - Build Command: (leave empty)
   - Output Directory: (leave empty)
   - Install Command: (leave empty - Vercel auto-installs from requirements.txt)

4. **Deploy**
   - Click **"Deploy"**
   - Wait 3-5 minutes
   - **Watch the build logs** for any errors

5. **Get Your URL**
   - After deployment, Vercel shows your URL
   - Format: `https://your-project-name.vercel.app`
   - **Copy this URL!**

---

## ✅ Step 3: Verify Deployment (1 minute)

1. **Visit Your URL**
   - Open: `https://your-project-name.vercel.app`
   - You should see the web interface or API docs

2. **Check Health**
   - Visit: `https://your-project-name.vercel.app/health`
   - Should return: `{"status": "healthy", ...}`

3. **Check API Docs**
   - Visit: `https://your-project-name.vercel.app/docs`
   - Should show FastAPI documentation

4. **Test Detection**
   - Upload an image
   - Click "Detect Deepfake"
   - Should see results

---

## ⚠️ If Deployment Fails

### Error: "Package size too large"
**This is expected! PyTorch is too large for Vercel.**

**Solution: Use Railway instead**
1. Go to https://railway.app
2. Login with GitHub
3. Click "New Project" → "Deploy from GitHub repo"
4. Select your repository
5. Railway auto-deploys (no size limits!)
6. Get your deployment URL in 2-3 minutes

### Error: "Import Error"
**Solution:**
1. Check Vercel build logs
2. Verify `requirements.txt` has all dependencies
3. Check that `api/index.py` exists
4. Verify `vercel.json` is correct

### Error: "Function Timeout"
**Solution:**
1. This is normal for ML inference
2. Consider optimizing your model
3. Use Railway (better for ML apps)

---

## 🎯 Alternative: Railway Deployment (Recommended)

**Railway is better for ML applications:**

1. **Go to Railway**
   - Visit: https://railway.app
   - Click "Start a New Project"

2. **Login with GitHub**
   - Authorize Railway

3. **Deploy from GitHub**
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Find: `dimpalmang20/Deepfake_Tool`
   - Click "Deploy"

4. **Wait 2-3 Minutes**
   - Railway builds automatically
   - No configuration needed!

5. **Get Your URL**
   - Click on your project
   - Go to "Settings" → "Networking"
   - Click "Generate Domain"
   - **Your deployment URL is ready!**

**Railway Advantages:**
- ✅ No size limits
- ✅ Better Python support
- ✅ Easier deployment
- ✅ Free tier available
- ✅ Auto-detects Python projects

---

## 📋 What Was Fixed

✅ **Python 3.12 Compatibility**
- Updated torch to 2.2.0+
- Updated torchvision to 0.17.0+
- Updated numpy to 1.26.0+

✅ **Vercel Configuration**
- Created `vercel.json`
- Created `api/index.py` handler
- Fixed file paths

✅ **Code Fixes**
- Fixed static file serving
- Added fallback HTML
- Improved error handling

---

## 🔗 Quick Links

- **Vercel Dashboard**: https://vercel.com/dashboard
- **Railway Dashboard**: https://railway.app/dashboard
- **GitHub Repo**: https://github.com/dimpalmang20/Deepfake_Tool
- **Deployment Helper**: Run `python deploy_vercel.py`

---

## 📞 Need Help?

1. **Check deployment logs** in Vercel/Railway dashboard
2. **Run deployment helper**: `python deploy_vercel.py`
3. **Read detailed guide**: `DEPLOY_VERCEL.md`
4. **Try Railway** if Vercel doesn't work (recommended!)

---

## 🎉 Success!

Once deployed, you'll have:
- ✅ Live deployment URL
- ✅ Working DeepFake Detection API
- ✅ Web interface
- ✅ API documentation at `/docs`

**Share your deployment link and start detecting deepfakes!** 🚀

---

## 💡 Pro Tips

1. **Use Railway instead of Vercel** for ML apps (no size limits!)
2. **Test locally first**: `python main.py`
3. **Monitor logs** during deployment
4. **Check `/health` endpoint** after deployment
5. **Use `/docs` endpoint** to test API

---

**Ready to deploy? Follow the 3 steps above!** 🚀

