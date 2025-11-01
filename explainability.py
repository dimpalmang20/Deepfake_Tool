"""
Explainability Module for DeepFake Detection

This module implements comprehensive explainability techniques for deepfake detection,
with a focus on Grad-CAM (Gradient-weighted Class Activation Mapping) to provide
transparent insights into model decision-making processes.

The explainability system generates heatmaps that highlight regions of the input
that most influenced the model's decision, enabling users to understand why
the model classified content as real or fake. This is crucial for building
trust in AI systems and for forensic analysis of manipulated content.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import os
from typing import List, Tuple, Dict, Optional, Union
from pathlib import Path
import json
from datetime import datetime


class GradCAMExplainer:
    """
    Comprehensive Grad-CAM explainer for deepfake detection models.
    
    This class implements advanced explainability techniques including:
    - Gradient-weighted Class Activation Mapping (Grad-CAM)
    - Guided Grad-CAM for enhanced visualization
    - Multi-layer attention visualization
    - Temporal attention for video analysis
    - Frequency domain attention mapping
    """
    
    def __init__(self, model: nn.Module, target_layers: Optional[List[str]] = None):
        """
        Initialize the Grad-CAM explainer.
        
        Args:
            model: Trained deepfake detection model
            target_layers: List of layer names to generate Grad-CAM for
        """
        self.model = model
        self.model.eval()
        
        # Default target layers for different backbones
        if target_layers is None:
            if hasattr(model, 'backbone_name'):
                if model.backbone_name == 'xception':
                    self.target_layers = ['backbone.conv4', 'backbone.conv3']
                elif model.backbone_name == 'efficientnet':
                    self.target_layers = ['backbone.features.6', 'backbone.features.4']
                else:
                    self.target_layers = ['backbone']
            else:
                self.target_layers = ['backbone']
        else:
            self.target_layers = target_layers
        
        # Register hooks for gradient computation
        self.gradients = {}
        self.activations = {}
        self.hooks = []
        
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks for gradient computation."""
        def forward_hook(module, input, output, name):
            self.activations[name] = output.detach()
        
        def backward_hook(module, grad_input, grad_output, name):
            self.gradients[name] = grad_output[0].detach()
        
        # Register hooks for target layers
        for name, module in self.model.named_modules():
            if any(target in name for target in self.target_layers):
                hook_f = lambda m, i, o, n=name: forward_hook(m, i, o, n)
                hook_b = lambda m, gi, go, n=name: backward_hook(m, gi, go, n)
                
                self.hooks.append(module.register_forward_hook(hook_f))
                self.hooks.append(module.register_backward_hook(hook_b))
    
    def _generate_gradcam(self, input_tensor: torch.Tensor, class_idx: int = None) -> Dict[str, np.ndarray]:
        """
        Generate Grad-CAM heatmaps for the input.
        
        Args:
            input_tensor: Input tensor (B, C, H, W) or (B, T, C, H, W)
            class_idx: Target class index (None for predicted class)
            
        Returns:
            Dictionary of Grad-CAM heatmaps for each target layer
        """
        # Forward pass
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        output[0, class_idx].backward(retain_graph=True)
        
        gradcam_maps = {}
        
        for layer_name in self.target_layers:
            if layer_name in self.activations and layer_name in self.gradients:
                # Get activations and gradients
                activations = self.activations[layer_name]
                gradients = self.gradients[layer_name]
                
                # Compute global average pooling of gradients
                weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
                
                # Generate Grad-CAM
                gradcam = torch.sum(weights * activations, dim=1, keepdim=True)
                gradcam = F.relu(gradcam)
                
                # Normalize and convert to numpy
                gradcam = gradcam.squeeze().cpu().numpy()
                gradcam = (gradcam - gradcam.min()) / (gradcam.max() - gradcam.min() + 1e-8)
                
                gradcam_maps[layer_name] = gradcam
        
        return gradcam_maps
    
    def _generate_guided_gradcam(self, input_tensor: torch.Tensor, class_idx: int = None) -> Dict[str, np.ndarray]:
        """
        Generate Guided Grad-CAM for enhanced visualization.
        
        Args:
            input_tensor: Input tensor
            class_idx: Target class index
            
        Returns:
            Dictionary of Guided Grad-CAM heatmaps
        """
        # Generate standard Grad-CAM
        gradcam_maps = self._generate_gradcam(input_tensor, class_idx)
        
        # Compute gradients for guided backpropagation
        input_tensor.requires_grad_(True)
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        output[0, class_idx].backward(retain_graph=True)
        
        # Get input gradients
        input_gradients = input_tensor.grad.data.cpu().numpy()
        
        guided_gradcam_maps = {}
        
        for layer_name, gradcam in gradcam_maps.items():
            # Resize Grad-CAM to input size
            gradcam_resized = cv2.resize(gradcam, (input_tensor.shape[-1], input_tensor.shape[-2]))
            
            # Apply guided backpropagation
            guided_gradcam = gradcam_resized * input_gradients[0].mean(axis=0)
            guided_gradcam = np.maximum(guided_gradcam, 0)  # ReLU
            guided_gradcam = (guided_gradcam - guided_gradcam.min()) / (guided_gradcam.max() - guided_gradcam.min() + 1e-8)
            
            guided_gradcam_maps[layer_name] = guided_gradcam
        
        return guided_gradcam_maps
    
    def _generate_temporal_attention(self, video_tensor: torch.Tensor, class_idx: int = None) -> np.ndarray:
        """
        Generate temporal attention map for video analysis.
        
        Args:
            video_tensor: Video tensor (B, T, C, H, W)
            class_idx: Target class index
            
        Returns:
            Temporal attention weights
        """
        batch_size, num_frames, channels, height, width = video_tensor.shape
        
        # Process each frame
        frame_attentions = []
        
        for t in range(num_frames):
            frame = video_tensor[:, t, :, :, :]
            
            # Generate Grad-CAM for this frame
            frame_gradcam = self._generate_gradcam(frame, class_idx)
            
            # Average across layers
            if frame_gradcam:
                avg_attention = np.mean(list(frame_gradcam.values()), axis=0)
                frame_attentions.append(avg_attention)
            else:
                frame_attentions.append(np.zeros((height, width)))
        
        return np.array(frame_attentions)
    
    def generate_explanation(self, 
                            input_tensor: torch.Tensor,
                            class_idx: int = None,
                            include_guided: bool = True,
                            include_temporal: bool = False) -> Dict[str, any]:
        """
        Generate comprehensive explanation for the input.
        
        Args:
            input_tensor: Input tensor (image or video)
            class_idx: Target class index
            include_guided: Whether to include guided Grad-CAM
            include_temporal: Whether to include temporal analysis
            
        Returns:
            Dictionary containing all explanation components
        """
        with torch.no_grad():
            # Get model prediction
            output = self.model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            
            if class_idx is None:
                class_idx = output.argmax(dim=1).item()
            
            prediction_confidence = probabilities[0, class_idx].item()
        
        # Generate Grad-CAM
        gradcam_maps = self._generate_gradcam(input_tensor, class_idx)
        
        explanation = {
            'prediction': class_idx,
            'confidence': prediction_confidence,
            'probabilities': probabilities[0].cpu().numpy(),
            'gradcam_maps': gradcam_maps
        }
        
        # Add guided Grad-CAM if requested
        if include_guided:
            guided_maps = self._generate_guided_gradcam(input_tensor, class_idx)
            explanation['guided_gradcam_maps'] = guided_maps
        
        # Add temporal analysis for videos
        if include_temporal and len(input_tensor.shape) == 5:
            temporal_attention = self._generate_temporal_attention(input_tensor, class_idx)
            explanation['temporal_attention'] = temporal_attention
        
        return explanation
    
    def visualize_explanation(self, 
                            input_tensor: torch.Tensor,
                            explanation: Dict[str, any],
                            save_path: Optional[str] = None,
                            figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Create comprehensive visualization of the explanation.
        
        Args:
            input_tensor: Original input tensor
            explanation: Explanation dictionary
            save_path: Optional path to save the visualization
            figsize: Figure size for the plot
            
        Returns:
            Matplotlib figure object
        """
        # Determine if input is video
        is_video = len(input_tensor.shape) == 5
        
        if is_video:
            # Use first frame for visualization
            input_image = input_tensor[0, 0].cpu().numpy().transpose(1, 2, 0)
            num_frames = input_tensor.shape[1]
        else:
            input_image = input_tensor[0].cpu().numpy().transpose(1, 2, 0)
            num_frames = 1
        
        # Denormalize image
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        input_image = input_image * std + mean
        input_image = np.clip(input_image, 0, 1)
        
        # Create subplot layout
        num_layers = len(explanation['gradcam_maps'])
        if explanation.get('guided_gradcam_maps'):
            num_layers += len(explanation['guided_gradcam_maps'])
        
        cols = min(4, num_layers + 1)  # +1 for original image
        rows = (num_layers + 1 + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        # Plot original image
        axes[0, 0].imshow(input_image)
        axes[0, 0].set_title(f'Original Image\nPrediction: {"Fake" if explanation["prediction"] else "Real"}\nConfidence: {explanation["confidence"]:.3f}')
        axes[0, 0.axis('off')
        
        # Plot Grad-CAM maps
        plot_idx = 1
        for layer_name, gradcam in explanation['gradcam_maps'].items():
            if plot_idx >= rows * cols:
                break
                
            row, col = divmod(plot_idx, cols)
            
            # Overlay heatmap on original image
            heatmap = cv2.applyColorMap(np.uint8(255 * gradcam), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            heatmap = cv2.resize(heatmap, (input_image.shape[1], input_image.shape[0]))
            
            overlay = cv2.addWeighted(
                (input_image * 255).astype(np.uint8), 0.6,
                heatmap, 0.4, 0
            )
            
            axes[row, col].imshow(overlay)
            axes[row, col].set_title(f'Grad-CAM: {layer_name}')
            axes[row, col].axis('off')
            
            plot_idx += 1
        
        # Plot guided Grad-CAM maps
        if 'guided_gradcam_maps' in explanation:
            for layer_name, guided_gradcam in explanation['guided_gradcam_maps'].items():
                if plot_idx >= rows * cols:
                    break
                    
                row, col = divmod(plot_idx, cols)
                
                # Overlay guided heatmap
                heatmap = cv2.applyColorMap(np.uint8(255 * guided_gradcam), cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                heatmap = cv2.resize(heatmap, (input_image.shape[1], input_image.shape[0]))
                
                overlay = cv2.addWeighted(
                    (input_image * 255).astype(np.uint8), 0.6,
                    heatmap, 0.4, 0
                )
                
                axes[row, col].imshow(overlay)
                axes[row, col].set_title(f'Guided Grad-CAM: {layer_name}')
                axes[row, col].axis('off')
                
                plot_idx += 1
        
        # Hide unused subplots
        for i in range(plot_idx, rows * cols):
            row, col = divmod(i, cols)
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Explanation visualization saved to: {save_path}")
        
        return fig
    
    def save_explanation(self, 
                        explanation: Dict[str, any],
                        save_dir: str,
                        filename: str = None) -> str:
        """
        Save explanation results to files.
        
        Args:
            explanation: Explanation dictionary
            save_dir: Directory to save files
            filename: Base filename (without extension)
            
        Returns:
            Path to saved explanation file
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"explanation_{timestamp}"
        
        # Save explanation metadata
        explanation_data = {
            'prediction': int(explanation['prediction']),
            'confidence': float(explanation['confidence']),
            'probabilities': explanation['probabilities'].tolist(),
            'timestamp': datetime.now().isoformat()
        }
        
        explanation_path = save_dir / f"{filename}.json"
        with open(explanation_path, 'w') as f:
            json.dump(explanation_data, f, indent=2)
        
        # Save heatmap images
        heatmap_dir = save_dir / f"{filename}_heatmaps"
        heatmap_dir.mkdir(exist_ok=True)
        
        for layer_name, gradcam in explanation['gradcam_maps'].items():
            # Save raw heatmap
            heatmap_path = heatmap_dir / f"gradcam_{layer_name}.png"
            plt.imsave(heatmap_path, gradcam, cmap='jet')
            
            # Save normalized heatmap
            normalized_gradcam = (gradcam - gradcam.min()) / (gradcam.max() - gradcam.min() + 1e-8)
            normalized_path = heatmap_dir / f"gradcam_{layer_name}_normalized.png"
            plt.imsave(normalized_path, normalized_gradcam, cmap='jet')
        
        # Save guided Grad-CAM if available
        if 'guided_gradcam_maps' in explanation:
            for layer_name, guided_gradcam in explanation['guided_gradcam_maps'].items():
                guided_path = heatmap_dir / f"guided_gradcam_{layer_name}.png"
                plt.imsave(guided_path, guided_gradcam, cmap='jet')
        
        print(f"Explanation saved to: {explanation_path}")
        return str(explanation_path)
    
    def generate_batch_explanations(self, 
                                  data_loader: torch.utils.data.DataLoader,
                                  num_samples: int = 10,
                                  save_dir: str = "outputs/explanations") -> List[Dict[str, any]]:
        """
        Generate explanations for a batch of samples.
        
        Args:
            data_loader: Data loader containing samples
            num_samples: Number of samples to process
            save_dir: Directory to save explanations
            
        Returns:
            List of explanation dictionaries
        """
        explanations = []
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        sample_count = 0
        for batch_idx, batch in enumerate(data_loader):
            if sample_count >= num_samples:
                break
            
            images = batch['image']
            labels = batch['label']
            
            for i in range(images.shape[0]):
                if sample_count >= num_samples:
                    break
                
                # Process single sample
                single_image = images[i:i+1]
                single_label = labels[i].item()
                
                # Generate explanation
                explanation = self.generate_explanation(
                    single_image,
                    class_idx=single_label,
                    include_guided=True,
                    include_temporal=len(single_image.shape) == 5
                )
                
                # Add metadata
                explanation['true_label'] = single_label
                explanation['sample_idx'] = sample_count
                
                # Save explanation
                filename = f"sample_{sample_count:04d}"
                self.save_explanation(explanation, save_dir, filename)
                
                explanations.append(explanation)
                sample_count += 1
        
        print(f"Generated {len(explanations)} explanations")
        return explanations
    
    def cleanup_hooks(self):
        """Remove all registered hooks to prevent memory leaks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.gradients.clear()
        self.activations.clear()


class AttentionVisualizer:
    """
    Advanced attention visualization for deepfake detection models.
    
    This class provides specialized visualization techniques for understanding
    how attention mechanisms focus on different regions of the input,
    particularly useful for analyzing temporal attention in video sequences.
    """
    
    def __init__(self, model: nn.Module):
        """
        Initialize attention visualizer.
        
        Args:
            model: Trained model with attention mechanisms
        """
        self.model = model
        self.attention_weights = {}
        self.hooks = []
    
    def _register_attention_hooks(self):
        """Register hooks to capture attention weights."""
        def attention_hook(module, input, output, name):
            if hasattr(module, 'attention_weights'):
                self.attention_weights[name] = module.attention_weights.detach()
        
        for name, module in self.model.named_modules():
            if 'attention' in name.lower():
                hook = module.register_forward_hook(
                    lambda m, i, o, n=name: attention_hook(m, i, o, n)
                )
                self.hooks.append(hook)
    
    def visualize_attention_flow(self, 
                              input_tensor: torch.Tensor,
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize attention flow across the network.
        
        Args:
            input_tensor: Input tensor
            save_path: Optional path to save visualization
            
        Returns:
            Matplotlib figure
        """
        # Register hooks
        self._register_attention_hooks()
        
        # Forward pass
        with torch.no_grad():
            output = self.model(input_tensor)
        
        # Create attention flow visualization
        num_attention_layers = len(self.attention_weights)
        if num_attention_layers == 0:
            print("No attention layers found in the model")
            return None
        
        fig, axes = plt.subplots(1, num_attention_layers, figsize=(5 * num_attention_layers, 5))
        if num_attention_layers == 1:
            axes = [axes]
        
        for idx, (layer_name, attention) in enumerate(self.attention_weights.items()):
            # Visualize attention weights
            if len(attention.shape) == 2:
                # 2D attention map
                im = axes[idx].imshow(attention.cpu().numpy(), cmap='viridis')
                axes[idx].set_title(f'Attention: {layer_name}')
                plt.colorbar(im, ax=axes[idx])
            else:
                # 1D attention weights
                axes[idx].plot(attention.cpu().numpy().flatten())
                axes[idx].set_title(f'Attention: {layer_name}')
                axes[idx].set_xlabel('Position')
                axes[idx].set_ylabel('Attention Weight')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Attention visualization saved to: {save_path}")
        
        return fig
    
    def cleanup_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.attention_weights.clear()


def demonstrate_explainability(model: nn.Module, 
                             sample_data: torch.Tensor,
                             save_dir: str = "outputs/explanations") -> Dict[str, any]:
    """
    Demonstrate explainability features with sample data.
    
    Args:
        model: Trained deepfake detection model
        sample_data: Sample input data
        save_dir: Directory to save results
        
    Returns:
        Dictionary containing demonstration results
    """
    print("Demonstrating explainability features...")
    
    # Initialize explainer
    explainer = GradCAMExplainer(model)
    
    # Generate explanation
    explanation = explainer.generate_explanation(
        sample_data,
        include_guided=True,
        include_temporal=len(sample_data.shape) == 5
    )
    
    # Create visualization
    fig = explainer.visualize_explanation(sample_data, explanation)
    
    # Save results
    save_path = explainer.save_explanation(explanation, save_dir)
    
    # Cleanup
    explainer.cleanup_hooks()
    
    print(f"Explainability demonstration completed")
    print(f"Prediction: {'Fake' if explanation['prediction'] else 'Real'}")
    print(f"Confidence: {explanation['confidence']:.3f}")
    print(f"Results saved to: {save_path}")
    
    return explanation


if __name__ == "__main__":
    print("DeepFake Detection Explainability Module")
    print("=" * 50)
    print("This module provides comprehensive explainability features")
    print("for deepfake detection models using Grad-CAM and attention visualization.")
    print("Ready for integration with training and inference pipelines.")

