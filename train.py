"""
Training Script for DeepFake Detection

This module implements a comprehensive training pipeline for deepfake detection
using state-of-the-art CNN architectures. The training process incorporates
advanced techniques including mixed precision training, heavy data augmentation,
frequency domain analysis, and explainability features.

The training pipeline is designed to handle large-scale datasets with thousands
of images and videos, implementing robust training strategies that capture
texture inconsistencies, frequency artifacts, and temporal irregularities
characteristic of deepfake content.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
# import wandb  # Optional: uncomment if using Weights & Biases
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Optional
import time
from datetime import datetime

from data_loader import create_data_loaders, FaceDataset
from utils.evaluation_metrics import DeepFakeEvaluator
from explainability import GradCAMExplainer


class DeepFakeDetector(nn.Module):
    """
    Advanced CNN-based deepfake detector with frequency domain analysis.
    
    This model implements a sophisticated architecture that combines:
    - Pretrained CNN backbone (Xception or EfficientNet) for feature extraction
    - Frequency domain analysis for manipulation artifact detection
    - Attention mechanisms for focusing on suspicious regions
    - Multi-scale feature fusion for comprehensive analysis
    """
    
    def __init__(self, 
                 backbone: str = 'xception',
                 num_classes: int = 2,
                 include_frequency_branch: bool = True,
                 dropout_rate: float = 0.5):
        """
        Initialize the deepfake detector.
        
        Args:
            backbone: CNN backbone ('xception' or 'efficientnet')
            num_classes: Number of output classes (2 for binary classification)
            include_frequency_branch: Whether to include frequency domain analysis
            dropout_rate: Dropout rate for regularization
        """
        super(DeepFakeDetector, self).__init__()
        
        self.backbone_name = backbone
        self.include_frequency_branch = include_frequency_branch
        
        # Load pretrained backbone for robust feature extraction
        if backbone == 'xception':
            self.backbone = models.xception(pretrained=True)
            # Remove the original classifier
            self.backbone.classifier = nn.Identity()
            backbone_features = 2048
        elif backbone == 'efficientnet':
            # Use EfficientNet-B4 for optimal performance
            from torchvision.models import efficientnet_b4
            self.backbone = efficientnet_b4(pretrained=True)
            self.backbone.classifier = nn.Identity()
            backbone_features = 1792
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Frequency domain analysis branch
        if include_frequency_branch:
            self.frequency_conv = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((7, 7)),
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
            )
            backbone_features += 512
        
        # Attention mechanism for focusing on suspicious regions
        self.attention = nn.Sequential(
            nn.Linear(backbone_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, backbone_features),
            nn.Sigmoid()
        )
        
        # Final classification head with advanced architecture
        self.classifier = nn.Sequential(
            nn.Linear(backbone_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights for optimal training
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights for optimal training convergence."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor, frequency_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor (B, C, H, W) or (B, T, C, H, W) for videos
            frequency_features: Optional frequency domain features
            
        Returns:
            Classification logits
        """
        batch_size = x.shape[0]
        
        # Handle video input by processing each frame
        if len(x.shape) == 5:  # Video input (B, T, C, H, W)
            # Reshape to process all frames
            x = x.view(-1, x.shape[2], x.shape[3], x.shape[4])  # (B*T, C, H, W)
            
            # Extract features from all frames
            features = self.backbone(x)  # (B*T, backbone_features)
            
            
            # Reshape back to (B, T, features)
            features = features.view(batch_size, -1, features.shape[1])
            
            # Temporal aggregation using attention
            # Compute attention weights for each frame
            attention_weights = self.attention(features.mean(dim=1))  # (B, features)
            attention_weights = attention_weights.unsqueeze(1)  # (B, 1, features)
            
            # Apply attention and aggregate temporally
            attended_features = features * attention_weights
            backbone_features = attended_features.mean(dim=1)  # (B, features)
        else:
            # Single image processing
            backbone_features = self.backbone(x)  # (B, backbone_features)
        
        # Frequency domain analysis
        if self.include_frequency_branch and frequency_features is not None:
            freq_features = self.frequency_conv(frequency_features)
            backbone_features = torch.cat([backbone_features, freq_features], dim=1)
        
        # Apply attention mechanism
        attention_weights = self.attention(backbone_features)
        attended_features = backbone_features * attention_weights
        
        # Final classification
        logits = self.classifier(attended_features)
        
        return logits


class FocalLoss(nn.Module):
    """
    Focal Loss implementation for handling class imbalance in deepfake detection.
    
    Focal Loss is particularly effective for deepfake detection as it focuses
    on hard examples and reduces the impact of easy examples, which is crucial
    when dealing with subtle manipulation artifacts.
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Model predictions (logits)
            targets: Ground truth labels
            
        Returns:
            Focal loss value
        """
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DeepFakeTrainer:
    """
    Comprehensive trainer for deepfake detection models.
    
    This trainer implements advanced training strategies including:
    - Mixed precision training for efficiency
    - Cosine annealing learning rate scheduling
    - Heavy data augmentation for robustness
    - Comprehensive evaluation metrics
    - Model checkpointing and early stopping
    """
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 test_loader: DataLoader,
                 device: str = 'cuda',
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-4,
                 use_focal_loss: bool = True,
                 use_mixed_precision: bool = True,
                 save_dir: str = 'outputs/models'):
        """
        Initialize the trainer.
        
        Args:
            model: Deepfake detection model
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            device: Training device
            learning_rate: Initial learning rate
            weight_decay: Weight decay for regularization
            use_focal_loss: Whether to use focal loss
            use_mixed_precision: Whether to use mixed precision training
            save_dir: Directory to save model checkpoints
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize optimizer with advanced settings
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Learning rate scheduler with cosine annealing
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,  # Restart period
            T_mult=2,  # Period multiplication factor
            eta_min=1e-7  # Minimum learning rate
        )
        
        # Loss function selection
        if use_focal_loss:
            self.criterion = FocalLoss(alpha=1.0, gamma=2.0)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Mixed precision training setup
        self.use_mixed_precision = use_mixed_precision
        self.scaler = GradScaler() if use_mixed_precision else None
        
        # Training state tracking
        self.best_val_auc = 0.0
        self.train_losses = []
        self.val_losses = []
        self.val_aucs = []
        self.learning_rates = []
        
        # Initialize evaluator for comprehensive metrics
        self.evaluator = DeepFakeEvaluator()
        
        # Initialize GradCAM explainer
        self.gradcam_explainer = GradCAMExplainer(self.model)
        
        print(f"Initialized trainer with {sum(p.numel() for p in self.model.parameters())} parameters")
        print(f"Using device: {device}")
        print(f"Mixed precision: {use_mixed_precision}")
        print(f"Focal loss: {use_focal_loss}")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch with comprehensive monitoring.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Handle video inputs
            if batch.get('is_video', False).any():
                # Process video frames
                batch_size = images.shape[0]
                num_frames = images.shape[1]
                images = images.view(-1, images.shape[2], images.shape[3], images.shape[4])
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.use_mixed_precision:
                with autocast():
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)
            else:
                logits = self.model(images)
                loss = self.criterion(logits, labels)
            
            # Backward pass with gradient scaling
            if self.use_mixed_precision:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            # Update learning rate
            self.scheduler.step()
            
            # Calculate metrics
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)
            
            total_loss += loss.item()
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities[:, 1].cpu().numpy())  # Fake class probability
            
            # Update progress bar
            current_lr = self.optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{correct_predictions/total_samples:.4f}',
                'LR': f'{current_lr:.2e}'
            })
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct_predictions / total_samples
        auc = roc_auc_score(all_labels, all_probabilities)
        
        # Store metrics
        self.train_losses.append(avg_loss)
        self.learning_rates.append(current_lr)
        
        metrics = {
            'train_loss': avg_loss,
            'train_accuracy': accuracy,
            'train_auc': auc,
            'learning_rate': current_lr
        }
        
        return metrics
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Validate for one epoch with comprehensive evaluation.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f'Validation {epoch}'):
                # Move batch to device
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Handle video inputs
                if batch.get('is_video', False).any():
                    batch_size = images.shape[0]
                    images = images.view(-1, images.shape[2], images.shape[3], images.shape[4])
                
                # Forward pass
                if self.use_mixed_precision:
                    with autocast():
                        logits = self.model(images)
                        loss = self.criterion(logits, labels)
                else:
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)
                
                # Calculate metrics
                probabilities = torch.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)
                
                total_loss += loss.item()
                correct_predictions += (predictions == labels).sum().item()
                total_samples += labels.size(0)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities[:, 1].cpu().numpy())
        
        # Calculate validation metrics
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct_predictions / total_samples
        auc = roc_auc_score(all_labels, all_probabilities)
        
        # Store metrics
        self.val_losses.append(avg_loss)
        self.val_aucs.append(auc)
        
        metrics = {
            'val_loss': avg_loss,
            'val_accuracy': accuracy,
            'val_auc': auc
        }
        
        return metrics
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """
        Save model checkpoint with comprehensive metadata.
        
        Args:
            epoch: Current epoch number
            metrics: Training metrics
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_aucs': self.val_aucs,
            'learning_rates': self.learning_rates
        }
        
        # Save regular checkpoint
        checkpoint_path = self.save_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.save_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"New best model saved with AUC: {metrics['val_auc']:.4f}")
    
    def train(self, num_epochs: int, early_stopping_patience: int = 10, 
              use_wandb: bool = False) -> Dict[str, List[float]]:
        """
        Complete training loop with comprehensive monitoring.
        
        Args:
            num_epochs: Number of epochs to train
            early_stopping_patience: Patience for early stopping
            use_wandb: Whether to use Weights & Biases logging
            
        Returns:
            Dictionary of training history
        """
        if use_wandb:
            wandb.init(project="deepfake-detection", config={
                'num_epochs': num_epochs,
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'weight_decay': self.optimizer.param_groups[0]['weight_decay'],
                'use_focal_loss': isinstance(self.criterion, FocalLoss),
                'use_mixed_precision': self.use_mixed_precision
            })
        
        best_auc = 0.0
        patience_counter = 0
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Early stopping patience: {early_stopping_patience}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)
            
            # Training phase
            train_metrics = self.train_epoch(epoch + 1)
            
            # Validation phase
            val_metrics = self.validate_epoch(epoch + 1)
            
            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics}
            
            # Log metrics
            print(f"Train Loss: {train_metrics['train_loss']:.4f}")
            print(f"Train Accuracy: {train_metrics['train_accuracy']:.4f}")
            print(f"Train AUC: {train_metrics['train_auc']:.4f}")
            print(f"Val Loss: {val_metrics['val_loss']:.4f}")
            print(f"Val Accuracy: {val_metrics['val_accuracy']:.4f}")
            print(f"Val AUC: {val_metrics['val_auc']:.4f}")
            print(f"Learning Rate: {train_metrics['learning_rate']:.2e}")
            
            # Check for best model
            is_best = val_metrics['val_auc'] > best_auc
            if is_best:
                best_auc = val_metrics['val_auc']
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Save checkpoint
            self.save_checkpoint(epoch + 1, epoch_metrics, is_best)
            
            # Log to wandb
            if use_wandb:
                wandb.log(epoch_metrics)
            
            # Early stopping check
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        # Final evaluation
        print("\nTraining completed!")
        print(f"Best validation AUC: {best_auc:.4f}")
        
        # Generate training plots
        self.plot_training_history()
        
        if use_wandb:
            wandb.finish()
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_aucs': self.val_aucs,
            'learning_rates': self.learning_rates
        }
    
    def plot_training_history(self):
        """Generate comprehensive training history plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        axes[0, 0].plot(self.train_losses, label='Train Loss', color='blue')
        axes[0, 0].plot(self.val_losses, label='Val Loss', color='red')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # AUC curves
        axes[0, 1].plot(self.val_aucs, label='Val AUC', color='green')
        axes[0, 1].set_title('Validation AUC')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('AUC')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate schedule
        axes[1, 0].plot(self.learning_rates, label='Learning Rate', color='orange')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Combined metrics
        axes[1, 1].plot(self.train_losses, label='Train Loss', color='blue', alpha=0.7)
        axes[1, 1].plot(self.val_losses, label='Val Loss', color='red', alpha=0.7)
        axes[1, 1].plot([x * max(self.val_losses) for x in self.val_aucs], 
                       label='Val AUC (scaled)', color='green', alpha=0.7)
        axes[1, 1].set_title('Combined Training Metrics')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Normalized Values')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training history plots saved to {self.save_dir / 'training_history.png'}")


def main():
    """Main training function with comprehensive setup."""
    parser = argparse.ArgumentParser(description='Train DeepFake Detection Model')
    parser.add_argument('--csv_file', type=str, required=True, help='Path to dataset CSV file')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--backbone', type=str, default='xception', choices=['xception', 'efficientnet'],
                       help='CNN backbone architecture')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for regularization')
    parser.add_argument('--use_focal_loss', action='store_true', help='Use focal loss for training')
    parser.add_argument('--use_mixed_precision', action='store_true', help='Use mixed precision training')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases logging')
    parser.add_argument('--save_dir', type=str, default='outputs/models', help='Directory to save models')
    
    args = parser.parse_args()
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        csv_file=args.csv_file,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=4
    )
    
    # Create model
    print(f"Creating model with {args.backbone} backbone...")
    model = DeepFakeDetector(
        backbone=args.backbone,
        num_classes=2,
        include_frequency_branch=True,
        dropout_rate=0.5
    )
    
    # Create trainer
    print("Initializing trainer...")
    trainer = DeepFakeTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        use_focal_loss=args.use_focal_loss,
        use_mixed_precision=args.use_mixed_precision,
        save_dir=args.save_dir
    )
    
    # Start training
    print("Starting training...")
    history = trainer.train(
        num_epochs=args.num_epochs,
        early_stopping_patience=15,
        use_wandb=args.use_wandb
    )
    
    print("Training completed successfully!")
    print(f"Model saved to: {args.save_dir}")


if __name__ == "__main__":
    main()

