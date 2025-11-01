"""
Evaluation Metrics for DeepFake Detection

This module provides comprehensive evaluation metrics for assessing
the performance of deepfake detection models. These metrics are crucial
for understanding model performance across different types of manipulations
and ensuring robust detection capabilities.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns


class DeepFakeEvaluator:
    """
    Comprehensive evaluator for deepfake detection models.
    
    This class provides methods to evaluate model performance using
    various metrics that are particularly relevant for deepfake detection,
    including temporal consistency for video analysis and explainability metrics.
    """
    
    def __init__(self):
        """Initialize the evaluator."""
        self.reset_metrics()
    
    def reset_metrics(self):
        """Reset all accumulated metrics."""
        self.predictions = []
        self.true_labels = []
        self.probabilities = []
        self.frame_scores = []  # For video analysis
        self.temporal_consistency = []
    
    def add_prediction(self, prediction: int, true_label: int, probability: float, 
                      frame_scores: Optional[List[float]] = None):
        """
        Add a single prediction to the evaluation.
        
        Args:
            prediction: Predicted class (0=real, 1=fake)
            true_label: True class label
            probability: Prediction probability
            frame_scores: List of frame-level scores (for video analysis)
        """
        self.predictions.append(prediction)
        self.true_labels.append(true_label)
        self.probabilities.append(probability)
        
        if frame_scores is not None:
            self.frame_scores.append(frame_scores)
            # Calculate temporal consistency as variance of frame scores
            temporal_var = np.var(frame_scores)
            self.temporal_consistency.append(temporal_var)
    
    def calculate_basic_metrics(self) -> Dict[str, float]:
        """
        Calculate basic classification metrics.
        
        Returns:
            Dictionary of basic metrics
        """
        if not self.predictions:
            return {}
        
        metrics = {
            'accuracy': accuracy_score(self.true_labels, self.predictions),
            'precision': precision_score(self.true_labels, self.predictions, average='weighted'),
            'recall': recall_score(self.true_labels, self.predictions, average='weighted'),
            'f1_score': f1_score(self.true_labels, self.predictions, average='weighted'),
            'auc_roc': roc_auc_score(self.true_labels, self.probabilities)
        }
        
        return metrics
    
    def calculate_temporal_metrics(self) -> Dict[str, float]:
        """
        Calculate temporal consistency metrics for video analysis.
        
        Returns:
            Dictionary of temporal metrics
        """
        if not self.frame_scores:
            return {}
        
        # Calculate average temporal consistency
        avg_temporal_consistency = np.mean(self.temporal_consistency)
        
        # Calculate frame-level accuracy
        frame_accuracies = []
        for frame_scores, true_label in zip(self.frame_scores, self.true_labels):
            frame_predictions = [1 if score > 0.5 else 0 for score in frame_scores]
            frame_accuracy = sum(1 for pred in frame_predictions if pred == true_label) / len(frame_predictions)
            frame_accuracies.append(frame_accuracy)
        
        metrics = {
            'avg_temporal_consistency': avg_temporal_consistency,
            'avg_frame_accuracy': np.mean(frame_accuracies),
            'frame_accuracy_std': np.std(frame_accuracies)
        }
        
        return metrics
    
    def calculate_explainability_metrics(self, heatmap_quality_scores: List[float]) -> Dict[str, float]:
        """
        Calculate metrics for explainability quality.
        
        Args:
            heatmap_quality_scores: List of quality scores for generated heatmaps
            
        Returns:
            Dictionary of explainability metrics
        """
        if not heatmap_quality_scores:
            return {}
        
        metrics = {
            'avg_heatmap_quality': np.mean(heatmap_quality_scores),
            'heatmap_quality_std': np.std(heatmap_quality_scores),
            'high_quality_heatmaps_ratio': sum(1 for score in heatmap_quality_scores if score > 0.7) / len(heatmap_quality_scores)
        }
        
        return metrics
    
    def generate_confusion_matrix(self, save_path: Optional[str] = None) -> np.ndarray:
        """
        Generate and optionally save confusion matrix.
        
        Args:
            save_path: Optional path to save the confusion matrix plot
            
        Returns:
            Confusion matrix as numpy array
        """
        if not self.predictions:
            return np.array([])
        
        cm = confusion_matrix(self.true_labels, self.predictions)
        
        if save_path:
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Real', 'Fake'],
                       yticklabels=['Real', 'Fake'])
            plt.title('Confusion Matrix - DeepFake Detection')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        return cm
    
    def generate_roc_curve(self, save_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate ROC curve data and optionally save plot.
        
        Args:
            save_path: Optional path to save the ROC curve plot
            
        Returns:
            Tuple of (fpr, tpr) arrays
        """
        if not self.probabilities:
            return np.array([]), np.array([])
        
        from sklearn.metrics import roc_curve
        
        fpr, tpr, _ = roc_curve(self.true_labels, self.probabilities)
        
        if save_path:
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_score(self.true_labels, self.probabilities):.3f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve - DeepFake Detection')
            plt.legend(loc="lower right")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        return fpr, tpr
    
    def generate_detailed_report(self) -> str:
        """
        Generate detailed classification report.
        
        Returns:
            Detailed classification report as string
        """
        if not self.predictions:
            return "No predictions available for evaluation."
        
        report = classification_report(
            self.true_labels, 
            self.predictions,
            target_names=['Real', 'Fake'],
            digits=4
        )
        
        return report
    
    def calculate_comprehensive_metrics(self, heatmap_quality_scores: Optional[List[float]] = None) -> Dict[str, any]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            heatmap_quality_scores: Optional list of heatmap quality scores
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        metrics = {}
        
        # Basic classification metrics
        metrics.update(self.calculate_basic_metrics())
        
        # Temporal metrics (if available)
        temporal_metrics = self.calculate_temporal_metrics()
        metrics.update(temporal_metrics)
        
        # Explainability metrics (if available)
        if heatmap_quality_scores:
            explainability_metrics = self.calculate_explainability_metrics(heatmap_quality_scores)
            metrics.update(explainability_metrics)
        
        # Add sample counts
        metrics['total_samples'] = len(self.predictions)
        metrics['real_samples'] = sum(1 for label in self.true_labels if label == 0)
        metrics['fake_samples'] = sum(1 for label in self.true_labels if label == 1)
        
        return metrics


def calculate_ensemble_metrics(individual_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Calculate ensemble metrics from multiple model evaluations.
    
    Args:
        individual_metrics: List of metrics dictionaries from individual models
        
    Returns:
        Dictionary of ensemble metrics
    """
    if not individual_metrics:
        return {}
    
    ensemble_metrics = {}
    
    # Calculate mean and std for each metric
    for metric_name in individual_metrics[0].keys():
        values = [metrics[metric_name] for metrics in individual_metrics if metric_name in metrics]
        if values:
            ensemble_metrics[f'{metric_name}_mean'] = np.mean(values)
            ensemble_metrics[f'{metric_name}_std'] = np.std(values)
    
    return ensemble_metrics


def evaluate_model_robustness(predictions: List[int], true_labels: List[int], 
                             noise_levels: List[float]) -> Dict[str, List[float]]:
    """
    Evaluate model robustness across different noise levels.
    
    Args:
        predictions: List of predictions
        true_labels: List of true labels
        noise_levels: List of noise levels tested
        
    Returns:
        Dictionary of robustness metrics
    """
    robustness_metrics = {
        'noise_levels': noise_levels,
        'accuracies': [],
        'f1_scores': []
    }
    
    for noise_level in noise_levels:
        # Simulate noise effect (in practice, this would be actual noisy predictions)
        noisy_predictions = predictions.copy()  # Placeholder for actual noise simulation
        
        accuracy = accuracy_score(true_labels, noisy_predictions)
        f1 = f1_score(true_labels, noisy_predictions, average='weighted')
        
        robustness_metrics['accuracies'].append(accuracy)
        robustness_metrics['f1_scores'].append(f1)
    
    return robustness_metrics

