"""
FastAPI Backend for DeepFake Detection

This module implements a comprehensive REST API for deepfake detection
with advanced features including real-time inference, explainability,
and batch processing. The API is designed for production deployment
with proper error handling, logging, and documentation.

The API provides endpoints for:
- Single image/video detection
- Batch processing
- Model health monitoring
- Explainability visualization
- Real-time streaming analysis
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import torch
import cv2
import numpy as np
import json
import os
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Union
import asyncio
import logging
from datetime import datetime
import base64
from PIL import Image
import io

from inference import DeepFakeInference
from explainability import GradCAMExplainer
from utils.face_detection import FaceDetector
from utils.evaluation_metrics import DeepFakeEvaluator


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="DeepFake Detection API",
    description="Advanced deepfake detection with explainability features",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
inference_system: Optional[DeepFakeInference] = None
model_loaded = False
processing_queue = []
results_cache = {}


class APIManager:
    """
    API Manager for handling inference requests and model management.
    
    This class manages the inference system, handles file processing,
    and provides caching mechanisms for efficient API responses.
    """
    
    def __init__(self):
        """Initialize the API manager."""
        self.inference_system = None
        self.model_loaded = False
        self.processing_queue = []
        self.results_cache = {}
        self.temp_dir = Path("temp")
        self.temp_dir.mkdir(exist_ok=True)
        
        # Static files directory for serving results
        self.static_dir = Path("outputs/api_static")
        self.static_dir.mkdir(parents=True, exist_ok=True)
    
    async def load_model(self, model_path: str, device: str = "auto"):
        """
        Load the deepfake detection model.
        
        Args:
            model_path: Path to trained model
            device: Device to use for inference
        """
        try:
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            
            self.inference_system = DeepFakeInference(
                model_path=model_path,
                device=device,
                confidence_threshold=0.5
            )
            
            self.model_loaded = True
            logger.info(f"Model loaded successfully on {device}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")
    
    async def process_image(self, 
                          image_data: bytes, 
                          filename: str,
                          generate_explanation: bool = True) -> Dict:
        """
        Process a single image for deepfake detection.
        
        Args:
            image_data: Image data as bytes
            filename: Original filename
            generate_explanation: Whether to generate Grad-CAM explanation
            
        Returns:
            Dictionary containing detection results
        """
        if not self.model_loaded:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        try:
            # Save temporary file
            temp_path = self.temp_dir / f"temp_{filename}"
            with open(temp_path, "wb") as f:
                f.write(image_data)
            
            # Process image
            result = self.inference_system.predict_image(
                str(temp_path),
                generate_explanation=generate_explanation,
                save_explanation=generate_explanation,
                output_dir=str(self.static_dir)
            )
            
            # Clean up temporary file
            temp_path.unlink()
            
            # Add API-specific metadata
            result['api_version'] = "1.0.0"
            result['processing_timestamp'] = datetime.now().isoformat()
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")
    
    async def process_video(self, 
                           video_data: bytes, 
                           filename: str,
                           generate_explanation: bool = True) -> Dict:
        """
        Process a single video for deepfake detection.
        
        Args:
            video_data: Video data as bytes
            filename: Original filename
            generate_explanation: Whether to generate temporal explanation
            
        Returns:
            Dictionary containing detection results
        """
        if not self.model_loaded:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        try:
            # Save temporary file
            temp_path = self.temp_dir / f"temp_{filename}"
            with open(temp_path, "wb") as f:
                f.write(video_data)
            
            # Process video
            result = self.inference_system.predict_video(
                str(temp_path),
                generate_explanation=generate_explanation,
                save_explanation=generate_explanation,
                output_dir=str(self.static_dir)
            )
            
            # Clean up temporary file
            temp_path.unlink()
            
            # Add API-specific metadata
            result['api_version'] = "1.0.0"
            result['processing_timestamp'] = datetime.now().isoformat()
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Video processing failed: {str(e)}")
    
    async def process_batch(self, 
                           files: List[UploadFile],
                           generate_explanations: bool = False) -> Dict:
        """
        Process a batch of files for deepfake detection.
        
        Args:
            files: List of uploaded files
            generate_explanations: Whether to generate explanations
            
        Returns:
            Dictionary containing batch processing results
        """
        if not self.model_loaded:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        results = []
        errors = []
        
        for file in files:
            try:
                # Read file data
                file_data = await file.read()
                
                # Determine file type
                is_video = file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv'))
                
                if is_video:
                    result = await self.process_video(
                        file_data, 
                        file.filename, 
                        generate_explanation=generate_explanations
                    )
                else:
                    result = await self.process_image(
                        file_data, 
                        file.filename, 
                        generate_explanation=generate_explanations
                    )
                
                results.append(result)
                
            except Exception as e:
                error_result = {
                    'filename': file.filename,
                    'error': str(e),
                    'prediction': 'error',
                    'confidence': 0.0
                }
                errors.append(error_result)
                logger.error(f"Error processing {file.filename}: {str(e)}")
        
        # Generate batch summary
        total_files = len(files)
        successful_predictions = len(results)
        error_count = len(errors)
        
        real_count = sum(1 for r in results if r.get('prediction') == 'real')
        fake_count = sum(1 for r in results if r.get('prediction') == 'fake')
        
        batch_summary = {
            'total_files': total_files,
            'successful_predictions': successful_predictions,
            'errors': error_count,
            'predictions': {
                'real': real_count,
                'fake': fake_count
            },
            'processing_timestamp': datetime.now().isoformat()
        }
        
        return {
            'summary': batch_summary,
            'results': results,
            'errors': errors
        }


# Initialize API manager
api_manager = APIManager()


@app.on_event("startup")
async def startup_event():
    """Initialize the API on startup."""
    logger.info("Starting DeepFake Detection API...")
    
    # Load model (you would set this path appropriately)
    model_path = "outputs/models/best_model.pth"  # Update with actual model path
    
    if os.path.exists(model_path):
        await api_manager.load_model(model_path)
        logger.info("API startup completed successfully")
    else:
        logger.warning(f"Model file not found at {model_path}. API will start without model.")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "DeepFake Detection API",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": api_manager.model_loaded,
        "endpoints": {
            "detect_image": "/detect/image",
            "detect_video": "/detect/video",
            "detect_batch": "/detect/batch",
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": api_manager.model_loaded,
        "timestamp": datetime.now().isoformat(),
        "system_info": {
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
    }


@app.post("/detect/image")
async def detect_image(
    file: UploadFile = File(...),
    generate_explanation: bool = True
):
    """
    Detect deepfake in a single image.
    
    Args:
        file: Uploaded image file
        generate_explanation: Whether to generate Grad-CAM explanation
        
    Returns:
        Detection results with optional explanation
    """
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Process image
    result = await api_manager.process_image(
        await file.read(),
        file.filename,
        generate_explanation=generate_explanation
    )
    
    return result


@app.post("/detect/video")
async def detect_video(
    file: UploadFile = File(...),
    generate_explanation: bool = True
):
    """
    Detect deepfake in a single video.
    
    Args:
        file: Uploaded video file
        generate_explanation: Whether to generate temporal explanation
        
    Returns:
        Detection results with optional explanation
    """
    # Validate file type
    if not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    # Process video
    result = await api_manager.process_video(
        await file.read(),
        file.filename,
        generate_explanation=generate_explanation
    )
    
    return result


@app.post("/detect/batch")
async def detect_batch(
    files: List[UploadFile] = File(...),
    generate_explanations: bool = False
):
    """
    Detect deepfake in multiple files.
    
    Args:
        files: List of uploaded files (images or videos)
        generate_explanations: Whether to generate explanations
        
    Returns:
        Batch processing results
    """
    if len(files) > 50:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum 50 files per batch")
    
    # Process batch
    result = await api_manager.process_batch(
        files,
        generate_explanations=generate_explanations
    )
    
    return result


@app.get("/explanation/{explanation_id}")
async def get_explanation(explanation_id: str):
    """
    Retrieve explanation visualization.
    
    Args:
        explanation_id: ID of the explanation
        
    Returns:
        Explanation visualization file
    """
    explanation_path = api_manager.static_dir / f"{explanation_id}_visualization.png"
    
    if not explanation_path.exists():
        raise HTTPException(status_code=404, detail="Explanation not found")
    
    return FileResponse(explanation_path)


@app.get("/model/info")
async def model_info():
    """Get model information and statistics."""
    if not api_manager.model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    model_info = {
        "model_loaded": True,
        "device": str(api_manager.inference_system.device),
        "confidence_threshold": api_manager.inference_system.confidence_threshold,
        "max_video_frames": api_manager.inference_system.max_video_frames,
        "model_architecture": "DeepFakeDetector",
        "features": [
            "Grad-CAM explainability",
            "Temporal analysis",
            "Frequency domain analysis",
            "Face detection and cropping",
            "Batch processing"
        ]
    }
    
    return model_info


@app.post("/model/reload")
async def reload_model(model_path: str, device: str = "auto"):
    """
    Reload the model with new parameters.
    
    Args:
        model_path: Path to new model file
        device: Device to use for inference
        
    Returns:
        Reload status
    """
    try:
        await api_manager.load_model(model_path, device)
        return {
            "status": "success",
            "message": f"Model reloaded from {model_path}",
            "device": device,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model reload failed: {str(e)}")


@app.get("/stats")
async def get_statistics():
    """Get API usage statistics."""
    # This would typically come from a database or logging system
    stats = {
        "total_requests": 0,  # Would be tracked in production
        "successful_predictions": 0,
        "failed_predictions": 0,
        "average_processing_time": 0.0,
        "model_uptime": "N/A",  # Would be calculated in production
        "timestamp": datetime.now().isoformat()
    }
    
    return stats


@app.get("/docs/custom")
async def custom_docs():
    """Custom API documentation."""
    return {
        "title": "DeepFake Detection API",
        "description": "Advanced deepfake detection with explainability features",
        "version": "1.0.0",
        "endpoints": {
            "POST /detect/image": {
                "description": "Detect deepfake in a single image",
                "parameters": {
                    "file": "Image file (JPEG, PNG, etc.)",
                    "generate_explanation": "Boolean to generate Grad-CAM explanation"
                },
                "response": {
                    "prediction": "real or fake",
                    "confidence": "Confidence score (0-1)",
                    "probabilities": "Real and fake probabilities",
                    "explanation": "Grad-CAM explanation if requested"
                }
            },
            "POST /detect/video": {
                "description": "Detect deepfake in a single video",
                "parameters": {
                    "file": "Video file (MP4, AVI, etc.)",
                    "generate_explanation": "Boolean to generate temporal explanation"
                },
                "response": {
                    "prediction": "real or fake",
                    "confidence": "Confidence score (0-1)",
                    "temporal_analysis": "Frame-by-frame analysis",
                    "explanation": "Temporal explanation if requested"
                }
            },
            "POST /detect/batch": {
                "description": "Detect deepfake in multiple files",
                "parameters": {
                    "files": "List of image/video files",
                    "generate_explanations": "Boolean to generate explanations"
                },
                "response": {
                    "summary": "Batch processing summary",
                    "results": "Individual file results",
                    "errors": "Processing errors if any"
                }
            }
        },
        "examples": {
            "curl_image": "curl -X POST 'http://localhost:8000/detect/image' -F 'file=@image.jpg' -F 'generate_explanation=true'",
            "curl_video": "curl -X POST 'http://localhost:8000/detect/video' -F 'file=@video.mp4' -F 'generate_explanation=true'",
            "curl_batch": "curl -X POST 'http://localhost:8000/detect/batch' -F 'files=@image1.jpg' -F 'files=@video1.mp4'"
        }
    }


# Mount static files for serving results
app.mount("/static", StaticFiles(directory="outputs/api_static"), name="static")


if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

