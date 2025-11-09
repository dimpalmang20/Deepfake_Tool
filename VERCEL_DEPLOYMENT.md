# Vercel Deployment Guide

## ✅ Fixed Issues

### Python 3.12 Compatibility
- Updated `torch` from `2.0.1` to `>=2.2.0` (Python 3.12 compatible)
- Updated `torchvision` from `0.15.2` to `>=0.17.0`
- Updated `numpy` from `1.24.3` to `>=1.26.0` (Python 3.12 compatible)
- Changed `opencv-python` to `opencv-python-headless` (better for serverless)

### Vercel Configuration
- Created `vercel.json` with proper Python runtime configuration
- Created `api/index.py` serverless function handler
- Configured function timeout and memory settings

## ⚠️ Important Considerations

### PyTorch Size Limitations
**PyTorch is very large (several GB)** and may exceed Vercel's serverless function size limits:
- **Hobby Plan**: 50MB limit
- **Pro Plan**: 250MB limit
- **Enterprise**: Custom limits

**PyTorch alone is much larger than these limits**, so deployment might fail or require:
1. **Vercel Pro/Enterprise plan** with larger limits
2. **Alternative deployment platforms** (Railway, Render, AWS Lambda with layers)
3. **Model optimization** (quantization, smaller models)
4. **External model storage** (load models from external URLs)

### Recommended Alternatives
If Vercel deployment fails due to size limits, consider:
1. **Railway.app** - No size limits, easier deployment
2. **Render.com** - Free tier with better Python support
3. **AWS Lambda** - With PyTorch layers
4. **Google Cloud Run** - Container-based, more flexible

## 🚀 Deployment Steps

1. **Install Vercel CLI** (if not already installed):
   ```bash
   npm i -g vercel
   ```

2. **Deploy from project directory**:
   ```bash
   cd DeepFake_Detector
   vercel
   ```

3. **Follow the prompts**:
   - Link to existing project or create new
   - Confirm settings
   - Wait for deployment

4. **Monitor deployment**:
   - Check build logs for any errors
   - If PyTorch size is an issue, consider alternatives above

## 📁 Files Modified for Vercel

- `requirements.txt` - Updated for Python 3.12 compatibility
- `vercel.json` - Vercel configuration
- `api/index.py` - Serverless function handler
- `VERCEL_DEPLOYMENT.md` - This file

## 🔧 Troubleshooting

### If deployment fails due to package size:
1. Consider using a lighter ML framework
2. Use external model hosting
3. Switch to Railway or Render for easier deployment

### If import errors occur:
1. Check that `api/index.py` correctly imports from `main.py`
2. Verify all dependencies are in `requirements.txt`
3. Check Vercel build logs for specific errors

## 📝 Notes

- The API handler in `api/index.py` imports the FastAPI app from `main.py`
- Static files (like `web_interface.html`) should be accessible from the project root
- Vercel automatically detects FastAPI apps and handles routing

## 🔗 Resources

- [Vercel Python Documentation](https://vercel.com/docs/concepts/functions/serverless-functions/runtimes/python)
- [PyTorch Installation](https://pytorch.org/get-started/locally/)
- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/)

