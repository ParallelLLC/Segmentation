#!/usr/bin/env python3
"""
Download SAM 2 Model Script

This script downloads the SAM 2 model checkpoints and sets up the environment
for few-shot and zero-shot segmentation experiments.
"""

import os
import sys
import requests
import zipfile
from pathlib import Path
import argparse
from tqdm import tqdm


def download_file(url: str, destination: str, chunk_size: int = 8192):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as file, tqdm(
        desc=os.path.basename(destination),
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            pbar.update(size)


def setup_sam2_environment():
    """Set up SAM 2 environment and download checkpoints."""
    print("Setting up SAM 2 environment...")
    
    # Create directories
    os.makedirs("models/checkpoints", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # SAM 2 model URLs (these are example URLs - replace with actual SAM 2 URLs)
    sam2_urls = {
        "vit_h": "https://dl.fbaipublicfiles.com/segment_anything_2/sam2_h.pth",
        "vit_l": "https://dl.fbaipublicfiles.com/segment_anything_2/sam2_l.pth",
        "vit_b": "https://dl.fbaipublicfiles.com/segment_anything_2/sam2_b.pth"
    }
    
    # Download SAM 2 checkpoints
    for model_name, url in sam2_urls.items():
        checkpoint_path = f"models/checkpoints/sam2_{model_name}.pth"
        
        if not os.path.exists(checkpoint_path):
            print(f"Downloading SAM 2 {model_name} checkpoint...")
            try:
                download_file(url, checkpoint_path)
                print(f"Successfully downloaded {model_name} checkpoint")
            except Exception as e:
                print(f"Failed to download {model_name} checkpoint: {e}")
                print("Please download manually from the SAM 2 repository")
        else:
            print(f"SAM 2 {model_name} checkpoint already exists")
    
    # Create symbolic links for easier access
    if not os.path.exists("sam2_checkpoint"):
        try:
            os.symlink("models/checkpoints/sam2_vit_h.pth", "sam2_checkpoint")
            print("Created symbolic link: sam2_checkpoint -> models/checkpoints/sam2_vit_h.pth")
        except:
            print("Could not create symbolic link (this is normal on Windows)")


def install_dependencies():
    """Install required dependencies."""
    print("Installing dependencies...")
    
    # Install from requirements.txt
    os.system("pip install -r requirements.txt")
    
    # Install SAM 2 specifically
    print("Installing SAM 2...")
    os.system("pip install git+https://github.com/facebookresearch/segment-anything-2.git")
    
    # Install CLIP
    print("Installing CLIP...")
    os.system("pip install git+https://github.com/openai/CLIP.git")


def create_demo_data():
    """Create demo data for testing."""
    print("Creating demo data...")
    
    # Create demo directories
    demo_dirs = [
        "data/satellite_demo",
        "data/fashion_demo", 
        "data/robotics_demo"
    ]
    
    for demo_dir in demo_dirs:
        os.makedirs(f"{demo_dir}/images", exist_ok=True)
        os.makedirs(f"{demo_dir}/masks", exist_ok=True)
    
    print("Demo data directories created. Run experiments to generate dummy data.")


def main():
    parser = argparse.ArgumentParser(description="Set up SAM 2 environment")
    parser.add_argument("--skip-download", action="store_true", 
                       help="Skip downloading SAM 2 checkpoints")
    parser.add_argument("--skip-install", action="store_true",
                       help="Skip installing dependencies")
    parser.add_argument("--demo-only", action="store_true",
                       help="Only create demo data directories")
    
    args = parser.parse_args()
    
    if args.demo_only:
        create_demo_data()
        return
    
    if not args.skip_install:
        install_dependencies()
    
    if not args.skip_download:
        setup_sam2_environment()
    
    create_demo_data()
    
    print("\nSetup complete!")
    print("\nNext steps:")
    print("1. Run few-shot satellite experiment:")
    print("   python experiments/few_shot_satellite.py --sam2_checkpoint sam2_checkpoint --data_dir data/satellite_demo")
    print("\n2. Run zero-shot fashion experiment:")
    print("   python experiments/zero_shot_fashion.py --sam2_checkpoint sam2_checkpoint --data_dir data/fashion_demo")
    print("\n3. Check the results/ directory for experiment outputs")


if __name__ == "__main__":
    main() 