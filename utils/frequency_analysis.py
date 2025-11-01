"""
Frequency Analysis Utilities for DeepFake Detection

This module implements frequency domain analysis techniques to detect
manipulation artifacts that are often invisible in the spatial domain.
Deepfake generation processes typically leave characteristic frequency
signatures that can be detected through DCT and high-pass filtering.
"""

import cv2
import numpy as np
from scipy import ndimage
from scipy.fft import dct, idct
from typing import Tuple, List
import torch
import torch.nn.functional as F


def generate_high_pass_filter(image: np.ndarray, cutoff_freq: float = 0.1) -> np.ndarray:
    """
    Generate high-pass filtered version of image to reveal manipulation artifacts.
    
    High-pass filtering emphasizes high-frequency components that are often
    altered during deepfake generation, revealing compression artifacts,
    blending inconsistencies, and frequency domain manipulation traces.
    
    Args:
        image: Input image (H, W, C)
        cutoff_freq: Cutoff frequency for high-pass filter (0.0 to 1.0)
        
    Returns:
        High-pass filtered image
    """
    # Convert to grayscale if color
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply Gaussian blur for low-pass component
    blurred = cv2.GaussianBlur(gray, (0, 0), 1/cutoff_freq)
    
    # High-pass = original - low-pass
    high_pass = gray.astype(np.float32) - blurred.astype(np.float32)
    
    # Normalize to 0-255 range
    high_pass = np.clip(high_pass + 128, 0, 255).astype(np.uint8)
    
    return high_pass


def generate_sobel_edges(image: np.ndarray) -> np.ndarray:
    """
    Generate Sobel edge map to detect texture inconsistencies.
    
    Sobel operators are particularly effective at detecting edges and
    texture patterns that may be inconsistent in deepfake images due
    to blending artifacts and unnatural texture synthesis.
    
    Args:
        image: Input image (H, W, C)
        
    Returns:
        Sobel edge magnitude map
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Sobel operators for x and y gradients
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Compute gradient magnitude
    sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Normalize to 0-255 range
    sobel_magnitude = np.clip(sobel_magnitude, 0, 255).astype(np.uint8)
    
    return sobel_magnitude


def generate_dct_coefficients(image: np.ndarray, block_size: int = 8) -> np.ndarray:
    """
    Generate DCT coefficients to analyze frequency domain characteristics.
    
    DCT (Discrete Cosine Transform) analysis reveals frequency domain
    patterns that are characteristic of different image generation methods.
    Deepfake images often show different DCT coefficient distributions
    compared to natural images.
    
    Args:
        image: Input image (H, W, C)
        block_size: Size of DCT blocks
        
    Returns:
        DCT coefficient map
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Ensure image dimensions are multiples of block_size
    h, w = gray.shape
    h_pad = ((h + block_size - 1) // block_size) * block_size
    w_pad = ((w + block_size - 1) // block_size) * block_size
    
    padded = np.zeros((h_pad, w_pad), dtype=np.float32)
    padded[:h, :w] = gray.astype(np.float32)
    
    # Apply DCT to each block
    dct_coeffs = np.zeros_like(padded)
    
    for i in range(0, h_pad, block_size):
        for j in range(0, w_pad, block_size):
            block = padded[i:i+block_size, j:j+block_size]
            dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
            dct_coeffs[i:i+block_size, j:j+block_size] = dct_block
    
    # Take absolute values and normalize
    dct_coeffs = np.abs(dct_coeffs)
    dct_coeffs = np.clip(dct_coeffs, 0, 255).astype(np.uint8)
    
    return dct_coeffs[:h, :w]


def generate_laplacian_variance(image: np.ndarray) -> float:
    """
    Compute Laplacian variance as a measure of image sharpness.
    
    Laplacian variance is a simple but effective measure of image
    sharpness. Deepfake images often have inconsistent sharpness
    patterns due to blending and synthesis artifacts.
    
    Args:
        image: Input image (H, W, C)
        
    Returns:
        Laplacian variance value
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply Laplacian operator
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    
    # Compute variance
    variance = laplacian.var()
    
    return variance


def generate_frequency_maps(image: np.ndarray) -> dict:
    """
    Generate comprehensive frequency analysis maps for deepfake detection.
    
    This function combines multiple frequency domain analysis techniques
    to create a comprehensive set of features for detecting manipulation
    artifacts that are characteristic of deepfake generation.
    
    Args:
        image: Input image (H, W, C)
        
    Returns:
        Dictionary containing various frequency analysis maps
    """
    frequency_maps = {}
    
    # High-pass filter to reveal high-frequency artifacts
    frequency_maps['high_pass'] = generate_high_pass_filter(image, cutoff_freq=0.1)
    
    # Sobel edges to detect texture inconsistencies
    frequency_maps['sobel_edges'] = generate_sobel_edges(image)
    
    # DCT coefficients for frequency domain analysis
    frequency_maps['dct_coeffs'] = generate_dct_coefficients(image, block_size=8)
    
    # Laplacian variance for sharpness analysis
    frequency_maps['laplacian_variance'] = generate_laplacian_variance(image)
    
    return frequency_maps


def create_frequency_visualization(frequency_maps: dict, image: np.ndarray) -> np.ndarray:
    """
    Create a comprehensive visualization of frequency analysis results.
    
    Args:
        frequency_maps: Dictionary of frequency analysis results
        image: Original image for reference
        
    Returns:
        Combined visualization image
    """
    # Resize all maps to same size
    target_size = (224, 224)
    
    # Original image
    orig_resized = cv2.resize(image, target_size)
    if len(orig_resized.shape) == 3:
        orig_gray = cv2.cvtColor(orig_resized, cv2.COLOR_BGR2GRAY)
    else:
        orig_gray = orig_resized
    
    # Resize frequency maps
    high_pass_resized = cv2.resize(frequency_maps['high_pass'], target_size)
    sobel_resized = cv2.resize(frequency_maps['sobel_edges'], target_size)
    dct_resized = cv2.resize(frequency_maps['dct_coeffs'], target_size)
    
    # Create 2x2 grid visualization
    grid_size = 224 * 2
    visualization = np.zeros((grid_size, grid_size), dtype=np.uint8)
    
    # Top-left: Original
    visualization[:224, :224] = orig_gray
    
    # Top-right: High-pass
    visualization[:224, 224:] = high_pass_resized
    
    # Bottom-left: Sobel edges
    visualization[224:, :224] = sobel_resized
    
    # Bottom-right: DCT coefficients
    visualization[224:, 224:] = dct_resized
    
    return visualization

