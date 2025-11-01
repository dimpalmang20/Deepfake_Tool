"""
Sample Dataset Creation Script

This script creates a sample dataset for demonstrating the deepfake detection system.
It generates synthetic images and videos to simulate a large-scale training dataset.
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import random

def create_sample_dataset(num_images=100, num_videos=20, output_dir="data"):
    """
    Create a sample dataset for demonstration.
    
    Args:
        num_images: Number of images to generate
        num_videos: Number of videos to generate
        output_dir: Output directory for the dataset
    """
    
    # Create directory structure
    os.makedirs(f"{output_dir}/images/real", exist_ok=True)
    os.makedirs(f"{output_dir}/images/fake", exist_ok=True)
    os.makedirs(f"{output_dir}/videos/real", exist_ok=True)
    os.makedirs(f"{output_dir}/videos/fake", exist_ok=True)
    
    dataset_entries = []
    
    print(f"🎨 Creating {num_images} sample images...")
    
    # Generate real images (simulated with natural patterns)
    for i in range(num_images // 2):
        # Create a more "natural" looking image
        image = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        
        # Add some structure to make it look more realistic
        # Add a face-like structure
        cv2.circle(image, (112, 100), 40, (180, 160, 140), -1)  # Face
        cv2.circle(image, (100, 90), 5, (0, 0, 0), -1)  # Left eye
        cv2.circle(image, (124, 90), 5, (0, 0, 0), -1)  # Right eye
        cv2.ellipse(image, (112, 110), (15, 8), 0, 0, 180, (0, 0, 0), 2)  # Mouth
        
        # Add some noise to make it look more realistic
        noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
        image = cv2.add(image, noise)
        
        image_path = f"{output_dir}/images/real/real_{i:03d}.jpg"
        cv2.imwrite(image_path, image)
        
        dataset_entries.append({
            'filepath': f"images/real/real_{i:03d}.jpg",
            'label': 0,  # Real
            'type': 'image'
        })
    
    # Generate fake images (with manipulation artifacts)
    for i in range(num_images // 2):
        # Create base image
        image = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        
        # Add face structure
        cv2.circle(image, (112, 100), 40, (180, 160, 140), -1)
        cv2.circle(image, (100, 90), 5, (0, 0, 0), -1)
        cv2.circle(image, (124, 90), 5, (0, 0, 0), -1)
        cv2.ellipse(image, (112, 110), (15, 8), 0, 0, 180, (0, 0, 0), 2)
        
        # Add "fake" characteristics - blur, compression artifacts
        image = cv2.GaussianBlur(image, (3, 3), 0)
        
        # Add compression-like artifacts
        image = cv2.resize(image, (112, 112))
        image = cv2.resize(image, (224, 224))
        
        # Add some unnatural patterns
        for _ in range(5):
            x, y = random.randint(0, 200), random.randint(0, 200)
            cv2.circle(image, (x, y), 3, (255, 255, 255), -1)
        
        image_path = f"{output_dir}/images/fake/fake_{i:03d}.jpg"
        cv2.imwrite(image_path, image)
        
        dataset_entries.append({
            'filepath': f"images/fake/fake_{i:03d}.jpg",
            'label': 1,  # Fake
            'type': 'image'
        })
    
    print(f"🎬 Creating {num_videos} sample videos...")
    
    # Generate real videos (simulated)
    for i in range(num_videos // 2):
        # Create a simple video with 10 frames
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_path = f"{output_dir}/videos/real/real_{i:03d}.mp4"
        out = cv2.VideoWriter(video_path, fourcc, 2.0, (224, 224))
        
        for frame_idx in range(10):
            # Create frame with slight movement
            frame = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
            
            # Add moving face
            center_x = 112 + int(5 * np.sin(frame_idx * 0.5))
            center_y = 100 + int(3 * np.cos(frame_idx * 0.3))
            
            cv2.circle(frame, (center_x, center_y), 40, (180, 160, 140), -1)
            cv2.circle(frame, (center_x-12, center_y-10), 5, (0, 0, 0), -1)
            cv2.circle(frame, (center_x+12, center_y-10), 5, (0, 0, 0), -1)
            cv2.ellipse(frame, (center_x, center_y+10), (15, 8), 0, 0, 180, (0, 0, 0), 2)
            
            out.write(frame)
        
        out.release()
        
        dataset_entries.append({
            'filepath': f"videos/real/real_{i:03d}.mp4",
            'label': 0,  # Real
            'type': 'video'
        })
    
    # Generate fake videos (with temporal inconsistencies)
    for i in range(num_videos // 2):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_path = f"{output_dir}/videos/fake/fake_{i:03d}.mp4"
        out = cv2.VideoWriter(video_path, fourcc, 2.0, (224, 224))
        
        for frame_idx in range(10):
            frame = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
            
            # Add inconsistent movement (fake characteristic)
            if frame_idx < 5:
                center_x = 112 + int(5 * np.sin(frame_idx * 0.5))
                center_y = 100 + int(3 * np.cos(frame_idx * 0.3))
            else:
                # Sudden jump (inconsistency)
                center_x = 112 + int(15 * np.sin(frame_idx * 0.5))
                center_y = 100 + int(10 * np.cos(frame_idx * 0.3))
            
            cv2.circle(frame, (center_x, center_y), 40, (180, 160, 140), -1)
            cv2.circle(frame, (center_x-12, center_y-10), 5, (0, 0, 0), -1)
            cv2.circle(frame, (center_x+12, center_y-10), 5, (0, 0, 0), -1)
            cv2.ellipse(frame, (center_x, center_y+10), (15, 8), 0, 0, 180, (0, 0, 0), 2)
            
            # Add fake artifacts
            if frame_idx % 3 == 0:
                frame = cv2.GaussianBlur(frame, (3, 3), 0)
            
            out.write(frame)
        
        out.release()
        
        dataset_entries.append({
            'filepath': f"videos/fake/fake_{i:03d}.mp4",
            'label': 1,  # Fake
            'type': 'video'
        })
    
    # Create CSV file
    df = pd.DataFrame(dataset_entries)
    csv_path = f"{output_dir}/dataset.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"\n✅ Dataset created successfully!")
    print(f"📁 Total samples: {len(dataset_entries)}")
    print(f"📊 Real samples: {len(df[df['label'] == 0])}")
    print(f"📊 Fake samples: {len(df[df['label'] == 1])}")
    print(f"🖼️ Images: {len(df[df['type'] == 'image'])}")
    print(f"🎬 Videos: {len(df[df['type'] == 'video'])}")
    print(f"💾 Dataset CSV: {csv_path}")
    
    return csv_path, df

if __name__ == "__main__":
    print("🎯 Creating Sample Dataset for DeepFake Detection")
    print("=" * 60)
    
    # Create sample dataset
    csv_path, dataset_df = create_sample_dataset(
        num_images=200,  # 100 real + 100 fake images
        num_videos=40,   # 20 real + 20 fake videos
        output_dir="data"
    )
    
    print(f"\n📋 Dataset Summary:")
    print(f"   CSV file: {csv_path}")
    print(f"   Total files: {len(dataset_df)}")
    print(f"   Real/Fake ratio: {len(dataset_df[df['label'] == 0]) / len(dataset_df[df['label'] == 1]):.2f}")
    
    print(f"\n🚀 Ready for training!")
    print(f"   Run: python train.py --csv_file {csv_path} --data_dir data/")





