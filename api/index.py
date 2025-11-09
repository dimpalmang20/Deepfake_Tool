"""
Vercel serverless function handler for FastAPI app

This file is the entry point for Vercel serverless functions.
It imports the FastAPI app from main.py and exports it for Vercel.
"""
import sys
import os
from pathlib import Path

# Get the project root directory (parent of api/)
project_root = Path(__file__).parent.parent.resolve()

# Add project root to Python path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Set working directory to project root for file access
os.chdir(project_root)

# Import the FastAPI app from main.py
try:
    from main import app
except ImportError as e:
    # If import fails, create a minimal error handler
    from fastapi import FastAPI
    app = FastAPI(title="DeepFake Detection System")
    
    @app.get("/")
    async def root():
        return {"error": f"Failed to import main app: {str(e)}", "message": "Please check deployment logs"}

# Export the app for Vercel
# Vercel's Python runtime automatically detects FastAPI/ASGI apps
# The handler variable is what Vercel looks for
handler = app

