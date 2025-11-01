"""
Inference Module for DeepFake Detection

This module provides comprehensive inference capabilities for real-time
deepfake detection, supporting both images and videos with advanced
explainability features. The inference system is optimized for production
deployment with efficient processing and detailed result reporting.

The inference pipeline includes:
- Real-time face detection and processing
- Temporal analysis for video sequences
- Grad-CAM explainability visualization
- Confidence scoring and uncertainty quantification
- Batch processing for efficiency
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import time
from datetime import datetime
import base64
from PIL import Image
import io

from data_loader import FaceDataset
from explainability import GradCAMExplainer
from utils.face_detection import FaceDetector
from utils.frequency_analysis import generate_frequency_maps
from utils.evaluation_metrics import DeepFakeEvaluator


class DeepFakeInference:
    """
    Comprehensive inference system for deepfake detection.
    
    This class provides production-ready inference capabilities with
    advanced features including temporal analysis, explainability,
    and confidence scoring for both images and videos.
    """
    
    def __init__(self, 
                 model_path: str,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 confidence_threshold: float = 0.5,
                 max_video_frames: int = 32):
        """
        Initialize the inference system.
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device for inference
            confidence_threshold: Threshold for binary classification
            max_video_frames: Maximum frames to process from videos
        """
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.max_video_frames = max_video_frames
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Initialize components
        self.face_detector = FaceDetector(device=device)
        self.gradcam_explainer = GradCAMExplainer(self.model)
        self.evaluator = DeepFakeEvaluator()
        
        print(f"Inference system initialized on {device}")
        print(f"Model loaded from: {model_path}")
        print(f"Confidence threshold: {confidence_threshold}")
    
    def _load_model(self, model_path: str) -> torch.nn.Module:
        """
        Load trained model from checkpoint.
        
        Args:
            model_path: Path to model checkpoint
            
        Returns:
            Loaded model
        """
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Extract model architecture info
        if 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
        else:
            model_state = checkpoint
        
        # Determine model architecture from state dict
        if any('backbone.conv4' in key for key in model_state.keys()):
            from train import DeepFakeDetector
            model = DeepFakeDetector(backbone='xception', num_classes=2)
        elif any('backbone.features.6' in key for key in model_state.keys()):
            from train import DeepFakeDetector
            model = DeepFakeDetector(backbone='efficientnet', num_classes=2)
        else:
            raise ValueError("Unable to determine model architecture from checkpoint")
        
        # Load state dict
        model.load_state_dict(model_state)
        model.to(self.device)
        
        return model
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for model inference.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed tensor
        """
        # Detect and crop faces
        face_crops = self.face_detector.extract_face_crops(image, target_size=(224, 224))
        
        if not face_crops:
            # Fallback to center crop
            h, w = image.shape[:2]
            center_crop = image[h//4:3*h//4, w//4:3*w//4]
            center_crop = cv2.resize(center_crop, (224, 224))
            face_crops = [center_crop]
        
        # Use the largest face
        face_crop = max(face_crops, key=lambda x: x.shape[0] * x.shape[1])
        
        # Convert to tensor and normalize
        face_tensor = torch.from_numpy(face_crop).permute(2, 0, 1).float() / 255.0
        
        # Normalize with ImageNet statistics
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        face_tensor = (face_tensor - mean) / std
        
        return face_tensor.unsqueeze(0).to(self.device)
    
    def _preprocess_video(self, video_path: str) -> Tuple[torch.Tensor, List[np.ndarray]]:
        """
        Preprocess video for model inference.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Tuple of (processed_tensor, frame_list)
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        
        while cap.isOpened() and frame_count < self.max_video_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract frames at regular intervals
            if frame_count % 3 == 0:  # Every 3rd frame
                # Detect and crop faces
                face_crops = self.face_detector.extract_face_crops(frame, target_size=(224, 224))
                
                if face_crops:
                    # Use largest face
                    face_crop = max(face_crops, key=lambda x: x.shape[0] * x.shape[1])
                    frames.append(face_crop)
                else:
                    # Fallback to center crop
                    h, w = frame.shape[:2]
                    center_crop = frame[h//4:3*h//4, w//4:3*w//4]
                    center_crop = cv2.resize(center_crop, (224, 224))
                    frames.append(center_crop)
            
            frame_count += 1
        
        cap.release()
        
        if not frames:
            # Create dummy frame if no frames extracted
            frames = [np.zeros((224, 224, 3), dtype=np.uint8)]
        
        # Convert to tensor
        frame_tensors = []
        for frame in frames:
            frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            
            # Normalize
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            frame_tensor = (frame_tensor - mean) / std
            
            frame_tensors.append(frame_tensor)
        
        # Stack frames
        video_tensor = torch.stack(frame_tensors).unsqueeze(0).to(self.device)
        
        return video_tensor, frames
    
    def _generate_frequency_features(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Generate frequency domain features for enhanced detection.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary of frequency features
        """
        return generate_frequency_maps(image)
    
    def predict_image(self, 
                     image_path: str,
                     generate_explanation: bool = True,
                     save_explanation: bool = True,
                     output_dir: str = "outputs/inference") -> Dict[str, any]:
        """
        Predict deepfake probability for a single image.
        
        Args:
            image_path: Path to image file
            generate_explanation: Whether to generate Grad-CAM explanation
            save_explanation: Whether to save explanation visualizations
            output_dir: Directory to save outputs
            
        Returns:
            Dictionary containing prediction results
        """
        start_time = time.time()
        
        # Load and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Preprocess for model
        input_tensor = self._preprocess_image(image)
        
        # Generate frequency features
        frequency_features = self._generate_frequency_features(image)
        
        # Model inference
        with torch.no_grad():
            logits = self.model(input_tensor)
            probabilities = F.softmax(logits, dim=1)
            prediction = torch.argmax(logits, dim=1).item()
            confidence = probabilities[0, prediction].item()
        
        # Generate explanation if requested
        explanation = None
        if generate_explanation:
            explanation = self.gradcam_explainer.generate_explanation(
                input_tensor,
                class_idx=prediction,
                include_guided=True
            )
            
            if save_explanation:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Save explanation
                filename = f"image_{Path(image_path).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                explanation_path = self.gradcam_explainer.save_explanation(
                    explanation, output_dir, filename
                )
                
                # Create visualization
                fig = self.gradcam_explainer.visualize_explanation(input_tensor, explanation)
                viz_path = output_dir / f"{filename}_visualization.png"
                fig.savefig(viz_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
        
        # Prepare results
        results = {
            'file_path': image_path,
            'file_type': 'image',
            'prediction': 'fake' if prediction == 1 else 'real',
            'confidence': confidence,
            'probabilities': {
                'real': probabilities[0, 0].item(),
                'fake': probabilities[0, 1].item()
            },
            'processing_time': time.time() - start_time,
            'timestamp': datetime.now().isoformat()
        }
        
        if explanation:
            results['explanation'] = {
                'gradcam_available': True,
                'explanation_path': explanation_path if save_explanation else None,
                'visualization_path': str(viz_path) if save_explanation else None
            }
        
        # Add frequency analysis
        results['frequency_analysis'] = {
            'high_pass_variance': float(np.var(frequency_features.get('high_pass', 0))),
            'sobel_edge_strength': float(np.mean(frequency_features.get('sobel_edges', 0))),
            'laplacian_variance': float(frequency_features.get('laplacian_variance', 0))
        }
        
        return results
    
    def predict_video(self, 
                     video_path: str,
                     generate_explanation: bool = True,
                     save_explanation: bool = True,
                     output_dir: str = "outputs/inference") -> Dict[str, any]:
        """
        Predict deepfake probability for a video.
        
        Args:
            video_path: Path to video file
            generate_explanation: Whether to generate temporal explanation
            save_explanation: Whether to save explanation visualizations
            output_dir: Directory to save outputs
            
        Returns:
            Dictionary containing prediction results
        """
        start_time = time.time()
        
        # Preprocess video
        video_tensor, frames = self._preprocess_video(video_path)
        
        # Model inference
        with torch.no_grad():
            logits = self.model(video_tensor)
            probabilities = F.softmax(logits, dim=1)
            prediction = torch.argmax(logits, dim=1).item()
            confidence = probabilities[0, prediction].item()
        
        # Frame-level analysis
        frame_scores = []
        frame_predictions = []
        
        for i, frame in enumerate(frames):
            # Process individual frame
            frame_tensor = self._preprocess_image(frame)
            
            with torch.no_grad():
                frame_logits = self.model(frame_tensor)
                frame_probs = F.softmax(frame_logits, dim=1)
                frame_pred = torch.argmax(frame_logits, dim=1).item()
                frame_conf = frame_probs[0, frame_pred].item()
            
            frame_scores.append(frame_conf)
            frame_predictions.append(frame_pred)
        
        # Temporal consistency analysis
        temporal_consistency = np.var(frame_scores)
        frame_agreement = sum(1 for p in frame_predictions if p == prediction) / len(frame_predictions)
        
        # Generate explanation if requested
        explanation = None
        if generate_explanation:
            explanation = self.gradcam_explainer.generate_explanation(
                video_tensor,
                class_idx=prediction,
                include_guided=True,
                include_temporal=True
            )
            
            if save_explanation:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Save explanation
                filename = f"video_{Path(video_path).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                explanation_path = self.gradcam_explainer.save_explanation(
                    explanation, output_dir, filename
                )
                
                # Create visualization
                fig = self.gradcam_explainer.visualize_explanation(video_tensor, explanation)
                viz_path = output_dir / f"{filename}_visualization.png"
                fig.savefig(viz_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
        
        # Prepare results
        results = {
            'file_path': video_path,
            'file_type': 'video',
            'prediction': 'fake' if prediction == 1 else 'real',
            'confidence': confidence,
            'probabilities': {
                'real': probabilities[0, 0].item(),
                'fake': probabilities[0, 1].item()
            },
            'temporal_analysis': {
                'num_frames_processed': len(frames),
                'frame_scores': frame_scores,
                'frame_predictions': frame_predictions,
                'temporal_consistency': float(temporal_consistency),
                'frame_agreement': float(frame_agreement)
            },
            'processing_time': time.time() - start_time,
            'timestamp': datetime.now().isoformat()
        }
        
        if explanation:
            results['explanation'] = {
                'gradcam_available': True,
                'temporal_attention_available': 'temporal_attention' in explanation,
                'explanation_path': explanation_path if save_explanation else None,
                'visualization_path': str(viz_path) if save_explanation else None
            }
        
        return results
    
    def predict_batch(self, 
                     file_paths: List[str],
                     output_dir: str = "outputs/inference",
                     generate_explanations: bool = False) -> List[Dict[str, any]]:
        """
        Predict deepfake probability for a batch of files.
        
        Args:
            file_paths: List of file paths (images or videos)
            output_dir: Directory to save outputs
            generate_explanations: Whether to generate explanations for all files
            
        Returns:
            List of prediction results
        """
        results = []
        
        for file_path in file_paths:
            try:
                # Determine file type
                is_video = file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv'))
                
                if is_video:
                    result = self.predict_video(
                        file_path,
                        generate_explanation=generate_explanations,
                        save_explanation=generate_explanations,
                        output_dir=output_dir
                    )
                else:
                    result = self.predict_image(
                        file_path,
                        generate_explanation=generate_explanations,
                        save_explanation=generate_explanations,
                        output_dir=output_dir
                    )
                
                results.append(result)
                
            except Exception as e:
                error_result = {
                    'file_path': file_path,
                    'error': str(e),
                    'prediction': 'error',
                    'confidence': 0.0
                }
                results.append(error_result)
                print(f"Error processing {file_path}: {str(e)}")
        
        return results
    
    def save_results(self, 
                    results: List[Dict[str, any]], 
                    output_path: str = "outputs/inference/results.json") -> str:
        """
        Save prediction results to JSON file.
        
        Args:
            results: List of prediction results
            output_path: Path to save results
            
        Returns:
            Path to saved results file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {output_path}")
        return str(output_path)
    
    def generate_summary_report(self, 
                              results: List[Dict[str, any]], 
                              output_path: str = "outputs/inference/summary_report.json") -> Dict[str, any]:
        """
        Generate comprehensive summary report.
        
        Args:
            results: List of prediction results
            output_path: Path to save summary report
            
        Returns:
            Summary report dictionary
        """
        # Calculate statistics
        total_files = len(results)
        real_predictions = sum(1 for r in results if r.get('prediction') == 'real')
        fake_predictions = sum(1 for r in results if r.get('prediction') == 'fake')
        error_predictions = sum(1 for r in results if r.get('prediction') == 'error')
        
        # Calculate average confidence
        valid_results = [r for r in results if 'confidence' in r and r['confidence'] > 0]
        avg_confidence = np.mean([r['confidence'] for r in valid_results]) if valid_results else 0
        
        # Calculate processing times
        processing_times = [r.get('processing_time', 0) for r in results if 'processing_time' in r]
        avg_processing_time = np.mean(processing_times) if processing_times else 0
        
        # Generate summary
        summary = {
            'total_files_processed': total_files,
            'predictions': {
                'real': real_predictions,
                'fake': fake_predictions,
                'errors': error_predictions
            },
            'confidence_statistics': {
                'average_confidence': float(avg_confidence),
                'min_confidence': float(min([r['confidence'] for r in valid_results], default=0)),
                'max_confidence': float(max([r['confidence'] for r in valid_results], default=0))
            },
            'performance_statistics': {
                'average_processing_time': float(avg_processing_time),
                'total_processing_time': float(sum(processing_times))
            },
            'file_type_breakdown': {
                'images': sum(1 for r in results if r.get('file_type') == 'image'),
                'videos': sum(1 for r in results if r.get('file_type') == 'video')
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Save summary
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Summary report saved to: {output_path}")
        return summary


def main():
    """Main inference function with command-line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description='DeepFake Detection Inference')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--input_path', type=str, required=True, help='Path to input file or directory')
    parser.add_argument('--output_dir', type=str, default='outputs/inference', help='Output directory')
    parser.add_argument('--generate_explanations', action='store_true', help='Generate Grad-CAM explanations')
    parser.add_argument('--confidence_threshold', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto, cpu, cuda)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # Initialize inference system
    print("Initializing inference system...")
    inference_system = DeepFakeInference(
        model_path=args.model_path,
        device=device,
        confidence_threshold=args.confidence_threshold
    )
    
    # Process input
    input_path = Path(args.input_path)
    
    if input_path.is_file():
        # Single file
        print(f"Processing single file: {input_path}")
        
        if input_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.flv']:
            result = inference_system.predict_video(
                str(input_path),
                generate_explanation=args.generate_explanations,
                save_explanation=args.generate_explanations,
                output_dir=args.output_dir
            )
        else:
            result = inference_system.predict_image(
                str(input_path),
                generate_explanation=args.generate_explanations,
                save_explanation=args.generate_explanations,
                output_dir=args.output_dir
            )
        
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Processing time: {result['processing_time']:.2f}s")
        
    elif input_path.is_dir():
        # Directory of files
        print(f"Processing directory: {input_path}")
        
        # Find all image and video files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv']
        
        file_paths = []
        for ext in image_extensions + video_extensions:
            file_paths.extend(input_path.glob(f"*{ext}"))
            file_paths.extend(input_path.glob(f"*{ext.upper()}"))
        
        print(f"Found {len(file_paths)} files to process")
        
        # Process batch
        results = inference_system.predict_batch(
            [str(p) for p in file_paths],
            output_dir=args.output_dir,
            generate_explanations=args.generate_explanations
        )
        
        # Save results
        results_path = inference_system.save_results(results, f"{args.output_dir}/results.json")
        
        # Generate summary
        summary = inference_system.generate_summary_report(results, f"{args.output_dir}/summary.json")
        
        print(f"Processed {len(results)} files")
        print(f"Results saved to: {results_path}")
        print(f"Summary: {summary['predictions']['real']} real, {summary['predictions']['fake']} fake")
    
    else:
        print(f"Invalid input path: {input_path}")
        return
    
    print("Inference completed successfully!")


if __name__ == "__main__":
    main()

