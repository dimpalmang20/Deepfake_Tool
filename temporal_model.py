"""
Temporal Model for Advanced Video DeepFake Detection

This module implements sophisticated temporal analysis models for detecting
deepfakes in video sequences. The temporal models capture frame-to-frame
inconsistencies, temporal artifacts, and motion patterns that are
characteristic of deepfake videos.

The temporal analysis includes:
- 3D CNN architectures for spatiotemporal feature extraction
- LSTM/GRU networks for temporal sequence modeling
- Attention mechanisms for temporal focus
- Motion analysis and optical flow features
- Temporal consistency scoring
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import math
from collections import OrderedDict


class TemporalAttention(nn.Module):
    """
    Temporal attention mechanism for focusing on suspicious frames.
    
    This module implements self-attention mechanisms to identify
    frames that are most likely to contain manipulation artifacts,
    enabling the model to focus computational resources on the most
    informative temporal regions.
    """
    
    def __init__(self, feature_dim: int, num_heads: int = 8, dropout: float = 0.1):
        """
        Initialize temporal attention module.
        
        Args:
            feature_dim: Dimension of input features
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(TemporalAttention, self).__init__()
        
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        
        assert feature_dim % num_heads == 0, "Feature dimension must be divisible by number of heads"
        
        # Linear projections for Q, K, V
        self.query_projection = nn.Linear(feature_dim, feature_dim)
        self.key_projection = nn.Linear(feature_dim, feature_dim)
        self.value_projection = nn.Linear(feature_dim, feature_dim)
        
        # Output projection
        self.output_projection = nn.Linear(feature_dim, feature_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(feature_dim)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through temporal attention.
        
        Args:
            x: Input tensor (batch_size, seq_len, feature_dim)
            mask: Optional attention mask
            
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size, seq_len, _ = x.shape
        
        # Apply layer normalization
        x_norm = self.layer_norm(x)
        
        # Compute Q, K, V
        Q = self.query_projection(x_norm)  # (batch_size, seq_len, feature_dim)
        K = self.key_projection(x_norm)     # (batch_size, seq_len, feature_dim)
        V = self.value_projection(x_norm)   # (batch_size, seq_len, feature_dim)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        
        # Reshape back
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.feature_dim
        )
        
        # Output projection
        output = self.output_projection(attended)
        
        # Residual connection
        output = output + x
        
        return output, attention_weights


class TemporalCNN3D(nn.Module):
    """
    3D CNN for spatiotemporal feature extraction from video sequences.
    
    This module implements a 3D CNN architecture that captures both
    spatial and temporal features simultaneously, making it particularly
    effective for detecting temporal inconsistencies in deepfake videos.
    """
    
    def __init__(self, 
                 input_channels: int = 3,
                 num_classes: int = 2,
                 dropout_rate: float = 0.5):
        """
        Initialize 3D CNN for temporal analysis.
        
        Args:
            input_channels: Number of input channels
            num_classes: Number of output classes
            dropout_rate: Dropout rate for regularization
        """
        super(TemporalCNN3D, self).__init__()
        
        # 3D Convolutional layers for spatiotemporal feature extraction
        self.conv3d_layers = nn.Sequential(
            # First 3D conv block
            nn.Conv3d(input_channels, 64, kernel_size=(3, 7, 7), padding=(1, 3, 3)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            
            # Second 3D conv block
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            
            # Third 3D conv block
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            
            # Fourth 3D conv block
            nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 7, 7))
        )
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through 3D CNN.
        
        Args:
            x: Input tensor (batch_size, channels, time, height, width)
            
        Returns:
            Classification logits
        """
        # Extract spatiotemporal features
        features = self.conv3d_layers(x)
        
        # Global pooling
        features = self.global_pool(features)
        features = features.view(features.size(0), -1)
        
        # Classification
        logits = self.classifier(features)
        
        return logits


class TemporalLSTM(nn.Module):
    """
    LSTM-based temporal model for sequence analysis.
    
    This module implements a bidirectional LSTM with attention mechanisms
    for analyzing temporal sequences in deepfake videos. The LSTM captures
    long-term dependencies and temporal patterns that are characteristic
    of manipulation artifacts.
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 512,
                 num_layers: int = 2,
                 num_classes: int = 2,
                 dropout_rate: float = 0.3):
        """
        Initialize temporal LSTM model.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Hidden dimension of LSTM
            num_layers: Number of LSTM layers
            num_classes: Number of output classes
            dropout_rate: Dropout rate
        """
        super(TemporalLSTM, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # Temporal attention
        self.temporal_attention = TemporalAttention(
            feature_dim=hidden_dim * 2,  # *2 for bidirectional
            num_heads=8,
            dropout=dropout_rate
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through temporal LSTM.
        
        Args:
            x: Input tensor (batch_size, seq_len, input_dim)
            lengths: Optional sequence lengths for padding
            
        Returns:
            Tuple of (logits, attention_weights)
        """
        batch_size, seq_len, _ = x.shape
        
        # Pack sequence if lengths provided
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Unpack sequence if packed
        if lengths is not None:
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        
        # Apply temporal attention
        attended_out, attention_weights = self.temporal_attention(lstm_out)
        
        # Global average pooling over time
        pooled_features = attended_out.mean(dim=1)  # (batch_size, hidden_dim * 2)
        
        # Classification
        logits = self.classifier(pooled_features)
        
        return logits, attention_weights


class HybridTemporalModel(nn.Module):
    """
    Hybrid temporal model combining 3D CNN and LSTM for comprehensive video analysis.
    
    This model combines the strengths of both 3D CNN (spatiotemporal feature extraction)
    and LSTM (temporal sequence modeling) to achieve robust deepfake detection
    in video sequences. The hybrid approach captures both local spatiotemporal
    patterns and long-term temporal dependencies.
    """
    
    def __init__(self,
                 input_channels: int = 3,
                 num_classes: int = 2,
                 lstm_hidden_dim: int = 512,
                 lstm_num_layers: int = 2,
                 dropout_rate: float = 0.3):
        """
        Initialize hybrid temporal model.
        
        Args:
            input_channels: Number of input channels
            num_classes: Number of output classes
            lstm_hidden_dim: Hidden dimension for LSTM
            lstm_num_layers: Number of LSTM layers
            dropout_rate: Dropout rate
        """
        super(HybridTemporalModel, self).__init__()
        
        # 3D CNN for spatiotemporal feature extraction
        self.temporal_cnn = TemporalCNN3D(
            input_channels=input_channels,
            num_classes=num_classes,
            dropout_rate=dropout_rate
        )
        
        # 2D CNN for frame-level feature extraction
        self.frame_cnn = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        
        # LSTM for temporal sequence modeling
        self.temporal_lstm = TemporalLSTM(
            input_dim=256 * 7 * 7,  # Output size of frame CNN
            hidden_dim=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            num_classes=num_classes,
            dropout_rate=dropout_rate
        )
        
        # Feature fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(512 + lstm_hidden_dim * 2, 1024),  # 3D CNN + LSTM features
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, num_classes)
        )
        
        # Attention weights for interpretability
        self.attention_weights = None
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through hybrid temporal model.
        
        Args:
            x: Input tensor (batch_size, channels, time, height, width)
            
        Returns:
            Tuple of (logits, attention_weights)
        """
        batch_size, channels, time, height, width = x.shape
        
        # 3D CNN path for spatiotemporal features
        cnn3d_features = self.temporal_cnn.conv3d_layers(x)
        cnn3d_pooled = self.temporal_cnn.global_pool(cnn3d_features)
        cnn3d_features = cnn3d_pooled.view(batch_size, -1)
        
        # 2D CNN + LSTM path for temporal sequence analysis
        # Reshape for frame-by-frame processing
        frames = x.view(batch_size * time, channels, height, width)
        
        # Extract frame features
        frame_features = self.frame_cnn(frames)
        frame_features = frame_features.view(batch_size, time, -1)
        
        # LSTM temporal analysis
        lstm_logits, attention_weights = self.temporal_lstm(frame_features)
        
        # Store attention weights for interpretability
        self.attention_weights = attention_weights
        
        # Combine features
        combined_features = torch.cat([cnn3d_features, lstm_logits], dim=1)
        
        # Final classification
        final_logits = self.fusion_layer(combined_features)
        
        return final_logits, attention_weights
    
    def get_temporal_attention(self) -> Optional[torch.Tensor]:
        """
        Get temporal attention weights for interpretability.
        
        Returns:
            Temporal attention weights if available
        """
        return self.attention_weights


class OpticalFlowExtractor:
    """
    Optical flow feature extractor for motion analysis in videos.
    
    This class implements optical flow computation and feature extraction
    to capture motion patterns that are characteristic of deepfake videos.
    Motion inconsistencies are often a strong indicator of manipulation.
    """
    
    def __init__(self):
        """Initialize optical flow extractor."""
        # Lucas-Kanade optical flow parameters
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # Farneback optical flow parameters
        self.farneback_params = dict(
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
    
    def extract_lucas_kanade_flow(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Extract Lucas-Kanade optical flow between consecutive frames.
        
        Args:
            frames: List of video frames
            
        Returns:
            List of optical flow fields
        """
        flows = []
        
        for i in range(len(frames) - 1):
            # Convert to grayscale
            prev_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)
            
            # Detect corners for tracking
            corners = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.3, minDistance=7)
            
            if corners is not None:
                # Track corners
                new_corners, status, error = cv2.calcOpticalFlowPyrLK(
                    prev_gray, curr_gray, corners, None, **self.lk_params
                )
                
                # Select good points
                good_corners = corners[status == 1]
                good_new_corners = new_corners[status == 1]
                
                # Compute flow vectors
                flow_vectors = good_new_corners - good_corners
                
                # Create flow field
                flow_field = np.zeros_like(prev_gray, dtype=np.float32)
                for (x, y), (dx, dy) in zip(good_corners.astype(int), flow_vectors):
                    if 0 <= x < flow_field.shape[1] and 0 <= y < flow_field.shape[0]:
                        flow_field[y, x] = np.sqrt(dx**2 + dy**2)
                
                flows.append(flow_field)
            else:
                flows.append(np.zeros_like(prev_gray, dtype=np.float32))
        
        return flows
    
    def extract_farneback_flow(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Extract Farneback dense optical flow between consecutive frames.
        
        Args:
            frames: List of video frames
            
        Returns:
            List of dense optical flow fields
        """
        flows = []
        
        for i in range(len(frames) - 1):
            # Convert to grayscale
            prev_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)
            
            # Compute dense optical flow
            flow = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, None, **self.farneback_params)
            
            # Compute flow magnitude
            flow_magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            flows.append(flow_magnitude)
        
        return flows
    
    def extract_motion_features(self, flows: List[np.ndarray]) -> Dict[str, float]:
        """
        Extract motion features from optical flow fields.
        
        Args:
            flows: List of optical flow fields
            
        Returns:
            Dictionary of motion features
        """
        if not flows:
            return {}
        
        # Concatenate all flows
        all_flows = np.concatenate([flow.flatten() for flow in flows])
        
        features = {
            'motion_mean': float(np.mean(all_flows)),
            'motion_std': float(np.std(all_flows)),
            'motion_max': float(np.max(all_flows)),
            'motion_median': float(np.median(all_flows)),
            'motion_entropy': float(self._compute_entropy(all_flows)),
            'temporal_consistency': float(self._compute_temporal_consistency(flows))
        }
        
        return features
    
    def _compute_entropy(self, data: np.ndarray) -> float:
        """Compute entropy of motion data."""
        hist, _ = np.histogram(data, bins=50, density=True)
        hist = hist[hist > 0]  # Remove zero bins
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        return entropy
    
    def _compute_temporal_consistency(self, flows: List[np.ndarray]) -> float:
        """Compute temporal consistency of motion patterns."""
        if len(flows) < 2:
            return 0.0
        
        # Compute correlation between consecutive flows
        correlations = []
        for i in range(len(flows) - 1):
            flow1 = flows[i].flatten()
            flow2 = flows[i + 1].flatten()
            
            if len(flow1) > 0 and len(flow2) > 0:
                correlation = np.corrcoef(flow1, flow2)[0, 1]
                if not np.isnan(correlation):
                    correlations.append(correlation)
        
        return float(np.mean(correlations)) if correlations else 0.0


class TemporalConsistencyAnalyzer:
    """
    Temporal consistency analyzer for detecting frame-to-frame inconsistencies.
    
    This class implements various temporal consistency measures that are
    particularly effective for detecting deepfake videos, as manipulation
    often introduces subtle temporal inconsistencies.
    """
    
    def __init__(self):
        """Initialize temporal consistency analyzer."""
        self.optical_flow_extractor = OpticalFlowExtractor()
    
    def analyze_temporal_consistency(self, frames: List[np.ndarray]) -> Dict[str, float]:
        """
        Analyze temporal consistency of video frames.
        
        Args:
            frames: List of video frames
            
        Returns:
            Dictionary of temporal consistency metrics
        """
        if len(frames) < 2:
            return {'error': 'Insufficient frames for temporal analysis'}
        
        # Extract optical flow
        lk_flows = self.optical_flow_extractor.extract_lucas_kanade_flow(frames)
        farneback_flows = self.optical_flow_extractor.extract_farneback_flow(frames)
        
        # Compute motion features
        lk_features = self.optical_flow_extractor.extract_motion_features(lk_flows)
        farneback_features = self.optical_flow_extractor.extract_motion_features(farneback_flows)
        
        # Compute frame similarity
        frame_similarities = self._compute_frame_similarities(frames)
        
        # Compute temporal gradients
        temporal_gradients = self._compute_temporal_gradients(frames)
        
        # Combine all metrics
        consistency_metrics = {
            'lk_motion_mean': lk_features.get('motion_mean', 0.0),
            'lk_motion_std': lk_features.get('motion_std', 0.0),
            'lk_temporal_consistency': lk_features.get('temporal_consistency', 0.0),
            'farneback_motion_mean': farneback_features.get('motion_mean', 0.0),
            'farneback_motion_std': farneback_features.get('motion_std', 0.0),
            'farneback_temporal_consistency': farneback_features.get('temporal_consistency', 0.0),
            'frame_similarity_mean': float(np.mean(frame_similarities)),
            'frame_similarity_std': float(np.std(frame_similarities)),
            'temporal_gradient_mean': float(np.mean(temporal_gradients)),
            'temporal_gradient_std': float(np.std(temporal_gradients))
        }
        
        return consistency_metrics
    
    def _compute_frame_similarities(self, frames: List[np.ndarray]) -> List[float]:
        """Compute similarity between consecutive frames."""
        similarities = []
        
        for i in range(len(frames) - 1):
            # Convert to grayscale
            frame1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            frame2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)
            
            # Compute structural similarity
            similarity = cv2.matchTemplate(frame1, frame2, cv2.TM_CCOEFF_NORMED)[0, 0]
            similarities.append(float(similarity))
        
        return similarities
    
    def _compute_temporal_gradients(self, frames: List[np.ndarray]) -> List[float]:
        """Compute temporal gradients between consecutive frames."""
        gradients = []
        
        for i in range(len(frames) - 1):
            # Convert to grayscale
            frame1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            frame2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)
            
            # Compute absolute difference
            diff = cv2.absdiff(frame1, frame2)
            gradient = np.mean(diff)
            gradients.append(float(gradient))
        
        return gradients


def create_temporal_model(model_type: str = 'hybrid',
                          input_channels: int = 3,
                          num_classes: int = 2,
                          **kwargs) -> nn.Module:
    """
    Create temporal model for video deepfake detection.
    
    Args:
        model_type: Type of temporal model ('3d_cnn', 'lstm', 'hybrid')
        input_channels: Number of input channels
        num_classes: Number of output classes
        **kwargs: Additional model parameters
        
    Returns:
        Temporal model instance
    """
    if model_type == '3d_cnn':
        return TemporalCNN3D(
            input_channels=input_channels,
            num_classes=num_classes,
            dropout_rate=kwargs.get('dropout_rate', 0.5)
        )
    
    elif model_type == 'lstm':
        return TemporalLSTM(
            input_dim=kwargs.get('input_dim', 256 * 7 * 7),
            hidden_dim=kwargs.get('hidden_dim', 512),
            num_layers=kwargs.get('num_layers', 2),
            num_classes=num_classes,
            dropout_rate=kwargs.get('dropout_rate', 0.3)
        )
    
    elif model_type == 'hybrid':
        return HybridTemporalModel(
            input_channels=input_channels,
            num_classes=num_classes,
            lstm_hidden_dim=kwargs.get('lstm_hidden_dim', 512),
            lstm_num_layers=kwargs.get('lstm_num_layers', 2),
            dropout_rate=kwargs.get('dropout_rate', 0.3)
        )
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


if __name__ == "__main__":
    print("Temporal Model for DeepFake Detection")
    print("=" * 50)
    print("This module provides advanced temporal analysis capabilities")
    print("for detecting deepfakes in video sequences.")
    print("Ready for integration with training and inference pipelines.")


