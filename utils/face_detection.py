"""
Face Detection Utilities for DeepFake Detection

This module provides face detection capabilities using MTCNN for robust
face localization in both images and videos. Face detection is crucial
for focusing the deepfake detection on facial regions where most
manipulations occur.
"""

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN
from PIL import Image
import os
from typing import List, Tuple, Optional


class FaceDetector:
    """
    Face detection wrapper using MTCNN for robust face localization.
    
    MTCNN (Multi-task CNN) is particularly effective for face detection
    in deepfake scenarios as it provides both face detection and
    landmark localization, which helps identify facial regions where
    manipulations are most likely to occur.
    """
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize face detector with MTCNN.
        
        Args:
            device: Device to run detection on ('cuda' or 'cpu')
        """
        self.device = device
        self.mtcnn = MTCNN(
            keep_all=True,
            device=device,
            min_face_size=40,  # Minimum face size to detect
            thresholds=[0.6, 0.7, 0.7]  # Detection thresholds for P-Net, R-Net, O-Net
        )
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in an image and return bounding boxes.
        
        Args:
            image: Input image as numpy array (H, W, C)
            
        Returns:
            List of bounding boxes as (x, y, width, height)
        """
        # Convert to PIL Image for MTCNN
        if len(image.shape) == 3 and image.shape[2] == 3:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = Image.fromarray(image)
        
        # Detect faces and landmarks
        boxes, probs, landmarks = self.mtcnn.detect(pil_image, landmarks=True)
        
        if boxes is None:
            return []
        
        # Convert to (x, y, width, height) format
        face_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = box
            face_boxes.append((int(x1), int(y1), int(x2-x1), int(y2-y1)))
        
        return face_boxes
    
    def extract_face_crops(self, image: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> List[np.ndarray]:
        """
        Extract and crop faces from image.
        
        Args:
            image: Input image as numpy array
            target_size: Target size for face crops (width, height)
            
        Returns:
            List of cropped face images
        """
        face_boxes = self.detect_faces(image)
        face_crops = []
        
        for x, y, w, h in face_boxes:
            # Extract face region
            face_crop = image[y:y+h, x:x+w]
            
            # Resize to target size
            face_crop = cv2.resize(face_crop, target_size)
            face_crops.append(face_crop)
        
        return face_crops
    
    def detect_faces_in_video(self, video_path: str, max_frames: int = 100) -> List[List[Tuple[int, int, int, int]]]:
        """
        Detect faces in video frames.
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to process
            
        Returns:
            List of face detections per frame
        """
        cap = cv2.VideoCapture(video_path)
        frame_detections = []
        frame_count = 0
        
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect faces in current frame
            face_boxes = self.detect_faces(frame)
            frame_detections.append(face_boxes)
            frame_count += 1
        
        cap.release()
        return frame_detections


def create_face_grid(face_crops: List[np.ndarray], grid_size: Tuple[int, int] = (3, 3)) -> np.ndarray:
    """
    Create a grid visualization of detected faces.
    
    Args:
        face_crops: List of face crop images
        grid_size: Grid dimensions (rows, cols)
        
    Returns:
        Grid image showing all detected faces
    """
    if not face_crops:
        return np.zeros((224, 224, 3), dtype=np.uint8)
    
    rows, cols = grid_size
    face_size = 224
    grid_image = np.zeros((rows * face_size, cols * face_size, 3), dtype=np.uint8)
    
    for i, face_crop in enumerate(face_crops[:rows * cols]):
        row = i // cols
        col = i % cols
        
        y_start = row * face_size
        y_end = (row + 1) * face_size
        x_start = col * face_size
        x_end = (col + 1) * face_size
        
        # Resize face crop to fit grid cell
        resized_face = cv2.resize(face_crop, (face_size, face_size))
        grid_image[y_start:y_end, x_start:x_end] = resized_face
    
    return grid_image

