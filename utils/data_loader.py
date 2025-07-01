"""
Data Loader Utilities

This module provides data loading utilities for different domains
(satellite, fashion, robotics) with support for few-shot and zero-shot learning.
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os
import json
from typing import List, Dict, Tuple, Optional
import random
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import cv2


class BaseDataLoader:
    """Base class for domain-specific data loaders."""
    
    def __init__(self, data_dir: str, image_size: Tuple[int, int] = (512, 512)):
        self.data_dir = data_dir
        self.image_size = image_size
        
        # Standard transforms
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.mask_transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])
    
    def load_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess image."""
        image = Image.open(image_path).convert('RGB')
        return self.transform(image)
    
    def load_mask(self, mask_path: str) -> torch.Tensor:
        """Load and preprocess mask."""
        mask = Image.open(mask_path).convert('L')
        return self.mask_transform(mask)
    
    def get_random_sample(self) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Get a random sample from the dataset."""
        raise NotImplementedError
    
    def get_class_examples(self, class_name: str, num_examples: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Get examples for a specific class."""
        raise NotImplementedError


class SatelliteDataLoader(BaseDataLoader):
    """Data loader for satellite imagery segmentation."""
    
    def __init__(self, data_dir: str, image_size: Tuple[int, int] = (512, 512)):
        super().__init__(data_dir, image_size)
        
        # Satellite-specific classes
        self.classes = ["building", "road", "vegetation", "water"]
        self.class_to_id = {cls: i for i, cls in enumerate(self.classes)}
        
        # Load dataset structure
        self.load_dataset_structure()
    
    def load_dataset_structure(self):
        """Load dataset structure and file paths."""
        self.images = []
        self.masks = []
        self.class_samples = {cls: [] for cls in self.classes}
        
        # Assuming structure: data_dir/images/ and data_dir/masks/
        images_dir = os.path.join(self.data_dir, "images")
        masks_dir = os.path.join(self.data_dir, "masks")
        
        if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
            # Create dummy data for demonstration
            self.create_dummy_data()
            return
        
        # Load real data
        for filename in os.listdir(images_dir):
            if filename.endswith(('.jpg', '.png', '.tif')):
                image_path = os.path.join(images_dir, filename)
                mask_path = os.path.join(masks_dir, filename.replace('.jpg', '_mask.png'))
                
                if os.path.exists(mask_path):
                    self.images.append(image_path)
                    self.masks.append(mask_path)
                    
                    # Categorize by class (simplified)
                    self.categorize_sample(image_path, mask_path)
    
    def create_dummy_data(self):
        """Create dummy satellite data for demonstration."""
        print("Creating dummy satellite data...")
        
        # Create dummy directory structure
        os.makedirs(os.path.join(self.data_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "masks"), exist_ok=True)
        
        # Generate dummy images and masks
        for i in range(100):
            # Create dummy image (satellite-like)
            image = np.random.randint(50, 200, (512, 512, 3), dtype=np.uint8)
            
            # Add some structure to make it look like satellite imagery
            # Buildings (rectangular shapes)
            for _ in range(5):
                x, y = np.random.randint(0, 400), np.random.randint(0, 400)
                w, h = np.random.randint(20, 80), np.random.randint(20, 80)
                image[y:y+h, x:x+w] = np.random.randint(100, 150, 3)
            
            # Roads (linear structures)
            for _ in range(3):
                x, y = np.random.randint(0, 512), np.random.randint(0, 512)
                length = np.random.randint(50, 150)
                angle = np.random.uniform(0, 2*np.pi)
                for j in range(length):
                    px = int(x + j * np.cos(angle))
                    py = int(y + j * np.sin(angle))
                    if 0 <= px < 512 and 0 <= py < 512:
                        image[py, px] = [80, 80, 80]
            
            # Save image
            image_path = os.path.join(self.data_dir, "images", f"satellite_{i:03d}.jpg")
            Image.fromarray(image).save(image_path)
            
            # Create corresponding mask
            mask = np.zeros((512, 512), dtype=np.uint8)
            
            # Add building masks
            for _ in range(3):
                x, y = np.random.randint(0, 400), np.random.randint(0, 400)
                w, h = np.random.randint(20, 80), np.random.randint(20, 80)
                mask[y:y+h, x:x+w] = 1  # Building class
            
            # Add road masks
            for _ in range(2):
                x, y = np.random.randint(0, 512), np.random.randint(0, 512)
                length = np.random.randint(50, 150)
                angle = np.random.uniform(0, 2*np.pi)
                for j in range(length):
                    px = int(x + j * np.cos(angle))
                    py = int(y + j * np.sin(angle))
                    if 0 <= px < 512 and 0 <= py < 512:
                        mask[py, px] = 2  # Road class
            
            # Save mask
            mask_path = os.path.join(self.data_dir, "masks", f"satellite_{i:03d}_mask.png")
            Image.fromarray(mask * 85).save(mask_path)  # Scale for visibility
            
            # Add to lists
            self.images.append(image_path)
            self.masks.append(mask_path)
            
            # Categorize
            self.categorize_sample(image_path, mask_path)
    
    def categorize_sample(self, image_path: str, mask_path: str):
        """Categorize sample by dominant class."""
        mask = np.array(Image.open(mask_path))
        
        # Count pixels for each class
        class_counts = {}
        for i, class_name in enumerate(self.classes):
            class_counts[class_name] = np.sum(mask == i)
        
        # Find dominant class
        dominant_class = max(class_counts.items(), key=lambda x: x[1])[0]
        self.class_samples[dominant_class].append((image_path, mask_path))
    
    def get_random_query(self, class_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a random query image and mask for a specific class."""
        if class_name not in self.class_samples or not self.class_samples[class_name]:
            # Fallback to any available sample
            idx = random.randint(0, len(self.images) - 1)
            image = self.load_image(self.images[idx])
            mask = self.load_mask(self.masks[idx])
            return image, mask
        
        # Get random sample from specified class
        image_path, mask_path = random.choice(self.class_samples[class_name])
        image = self.load_image(image_path)
        mask = self.load_mask(mask_path)
        
        return image, mask
    
    def get_class_examples(self, class_name: str, num_examples: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Get examples for a specific class."""
        examples = []
        
        if class_name in self.class_samples:
            available_samples = self.class_samples[class_name]
            selected_samples = random.sample(available_samples, min(num_examples, len(available_samples)))
            
            for image_path, mask_path in selected_samples:
                image = self.load_image(image_path)
                mask = self.load_mask(mask_path)
                examples.append((image, mask))
        
        return examples


class FashionDataLoader(BaseDataLoader):
    """Data loader for fashion segmentation."""
    
    def __init__(self, data_dir: str, image_size: Tuple[int, int] = (512, 512)):
        super().__init__(data_dir, image_size)
        
        # Fashion-specific classes
        self.classes = ["shirt", "pants", "dress", "shoes"]
        self.class_to_id = {cls: i for i, cls in enumerate(self.classes)}
        
        # Load dataset structure
        self.load_dataset_structure()
    
    def load_dataset_structure(self):
        """Load dataset structure and file paths."""
        self.images = []
        self.masks = []
        self.class_samples = {cls: [] for cls in self.classes}
        
        # Assuming structure: data_dir/images/ and data_dir/masks/
        images_dir = os.path.join(self.data_dir, "images")
        masks_dir = os.path.join(self.data_dir, "masks")
        
        if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
            # Create dummy data for demonstration
            self.create_dummy_data()
            return
        
        # Load real data
        for filename in os.listdir(images_dir):
            if filename.endswith(('.jpg', '.png')):
                image_path = os.path.join(images_dir, filename)
                mask_path = os.path.join(masks_dir, filename.replace('.jpg', '_mask.png'))
                
                if os.path.exists(mask_path):
                    self.images.append(image_path)
                    self.masks.append(mask_path)
                    
                    # Categorize by class
                    self.categorize_sample(image_path, mask_path)
    
    def create_dummy_data(self):
        """Create dummy fashion data for demonstration."""
        print("Creating dummy fashion data...")
        
        # Create dummy directory structure
        os.makedirs(os.path.join(self.data_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "masks"), exist_ok=True)
        
        # Generate dummy images and masks
        for i in range(100):
            # Create dummy image (fashion-like)
            image = np.random.randint(200, 255, (512, 512, 3), dtype=np.uint8)
            
            # Add fashion items
            class_id = i % len(self.classes)
            
            if class_id == 0:  # Shirt
                # Create shirt-like shape
                center_x, center_y = 256, 256
                width, height = 150, 200
                image[center_y-height//2:center_y+height//2, center_x-width//2:center_x+width//2] = [100, 150, 200]
            
            elif class_id == 1:  # Pants
                # Create pants-like shape
                center_x, center_y = 256, 300
                width, height = 120, 180
                image[center_y-height//2:center_y+height//2, center_x-width//2:center_x+width//2] = [50, 100, 150]
            
            elif class_id == 2:  # Dress
                # Create dress-like shape
                center_x, center_y = 256, 250
                width, height = 140, 220
                image[center_y-height//2:center_y+height//2, center_x-width//2:center_x+width//2] = [200, 100, 150]
            
            else:  # Shoes
                # Create shoes-like shape
                center_x, center_y = 256, 400
                width, height = 100, 60
                image[center_y-height//2:center_y+height//2, center_x-width//2:center_x+width//2] = [80, 80, 80]
            
            # Save image
            image_path = os.path.join(self.data_dir, "images", f"fashion_{i:03d}.jpg")
            Image.fromarray(image).save(image_path)
            
            # Create corresponding mask
            mask = np.zeros((512, 512), dtype=np.uint8)
            
            # Add mask for the fashion item
            if class_id == 0:  # Shirt
                center_x, center_y = 256, 256
                width, height = 150, 200
                mask[center_y-height//2:center_y+height//2, center_x-width//2:center_x+width//2] = 1
            
            elif class_id == 1:  # Pants
                center_x, center_y = 256, 300
                width, height = 120, 180
                mask[center_y-height//2:center_y+height//2, center_x-width//2:center_x+width//2] = 2
            
            elif class_id == 2:  # Dress
                center_x, center_y = 256, 250
                width, height = 140, 220
                mask[center_y-height//2:center_y+height//2, center_x-width//2:center_x+width//2] = 3
            
            else:  # Shoes
                center_x, center_y = 256, 400
                width, height = 100, 60
                mask[center_y-height//2:center_y+height//2, center_x-width//2:center_x+width//2] = 4
            
            # Save mask
            mask_path = os.path.join(self.data_dir, "masks", f"fashion_{i:03d}_mask.png")
            Image.fromarray(mask * 51).save(mask_path)  # Scale for visibility
            
            # Add to lists
            self.images.append(image_path)
            self.masks.append(mask_path)
            
            # Categorize
            self.categorize_sample(image_path, mask_path)
    
    def categorize_sample(self, image_path: str, mask_path: str):
        """Categorize sample by dominant class."""
        mask = np.array(Image.open(mask_path))
        
        # Count pixels for each class
        class_counts = {}
        for i, class_name in enumerate(self.classes):
            class_counts[class_name] = np.sum(mask == (i + 1))  # +1 because 0 is background
        
        # Find dominant class
        dominant_class = max(class_counts.items(), key=lambda x: x[1])[0]
        self.class_samples[dominant_class].append((image_path, mask_path))
    
    def get_test_sample(self) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Get a random test sample with ground truth masks."""
        idx = random.randint(0, len(self.images) - 1)
        image = self.load_image(self.images[idx])
        mask = self.load_mask(self.masks[idx])
        
        # Convert single mask to multi-class dictionary
        ground_truth = {}
        for i, class_name in enumerate(self.classes):
            class_mask = (mask == (i + 1)).float()  # +1 because 0 is background
            ground_truth[class_name] = class_mask
        
        return image, ground_truth


class RoboticsDataLoader(BaseDataLoader):
    """Data loader for robotics segmentation."""
    
    def __init__(self, data_dir: str, image_size: Tuple[int, int] = (512, 512)):
        super().__init__(data_dir, image_size)
        
        # Robotics-specific classes
        self.classes = ["robot", "tool", "safety"]
        self.class_to_id = {cls: i for i, cls in enumerate(self.classes)}
        
        # Load dataset structure
        self.load_dataset_structure()
    
    def load_dataset_structure(self):
        """Load dataset structure and file paths."""
        self.images = []
        self.masks = []
        self.class_samples = {cls: [] for cls in self.classes}
        
        # Assuming structure: data_dir/images/ and data_dir/masks/
        images_dir = os.path.join(self.data_dir, "images")
        masks_dir = os.path.join(self.data_dir, "masks")
        
        if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
            # Create dummy data for demonstration
            self.create_dummy_data()
            return
        
        # Load real data
        for filename in os.listdir(images_dir):
            if filename.endswith(('.jpg', '.png')):
                image_path = os.path.join(images_dir, filename)
                mask_path = os.path.join(masks_dir, filename.replace('.jpg', '_mask.png'))
                
                if os.path.exists(mask_path):
                    self.images.append(image_path)
                    self.masks.append(mask_path)
                    
                    # Categorize by class
                    self.categorize_sample(image_path, mask_path)
    
    def create_dummy_data(self):
        """Create dummy robotics data for demonstration."""
        print("Creating dummy robotics data...")
        
        # Create dummy directory structure
        os.makedirs(os.path.join(self.data_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "masks"), exist_ok=True)
        
        # Generate dummy images and masks
        for i in range(100):
            # Create dummy image (robotics-like)
            image = np.random.randint(50, 150, (512, 512, 3), dtype=np.uint8)
            
            # Add robotics elements
            class_id = i % len(self.classes)
            
            if class_id == 0:  # Robot
                # Create robot-like shape
                center_x, center_y = 256, 256
                width, height = 120, 160
                image[center_y-height//2:center_y+height//2, center_x-width//2:center_x+width//2] = [100, 100, 100]
            
            elif class_id == 1:  # Tool
                # Create tool-like shape
                center_x, center_y = 256, 256
                width, height = 80, 120
                image[center_y-height//2:center_y+height//2, center_x-width//2:center_x+width//2] = [150, 100, 50]
            
            else:  # Safety equipment
                # Create safety equipment-like shape
                center_x, center_y = 256, 256
                width, height = 100, 100
                image[center_y-height//2:center_y+height//2, center_x-width//2:center_x+width//2] = [200, 200, 50]
            
            # Save image
            image_path = os.path.join(self.data_dir, "images", f"robotics_{i:03d}.jpg")
            Image.fromarray(image).save(image_path)
            
            # Create corresponding mask
            mask = np.zeros((512, 512), dtype=np.uint8)
            
            # Add mask for the robotics element
            if class_id == 0:  # Robot
                center_x, center_y = 256, 256
                width, height = 120, 160
                mask[center_y-height//2:center_y+height//2, center_x-width//2:center_x+width//2] = 1
            
            elif class_id == 1:  # Tool
                center_x, center_y = 256, 256
                width, height = 80, 120
                mask[center_y-height//2:center_y+height//2, center_x-width//2:center_x+width//2] = 2
            
            else:  # Safety equipment
                center_x, center_y = 256, 256
                width, height = 100, 100
                mask[center_y-height//2:center_y+height//2, center_x-width//2:center_x+width//2] = 3
            
            # Save mask
            mask_path = os.path.join(self.data_dir, "masks", f"robotics_{i:03d}_mask.png")
            Image.fromarray(mask * 85).save(mask_path)  # Scale for visibility
            
            # Add to lists
            self.images.append(image_path)
            self.masks.append(mask_path)
            
            # Categorize
            self.categorize_sample(image_path, mask_path)
    
    def categorize_sample(self, image_path: str, mask_path: str):
        """Categorize sample by dominant class."""
        mask = np.array(Image.open(mask_path))
        
        # Count pixels for each class
        class_counts = {}
        for i, class_name in enumerate(self.classes):
            class_counts[class_name] = np.sum(mask == (i + 1))  # +1 because 0 is background
        
        # Find dominant class
        dominant_class = max(class_counts.items(), key=lambda x: x[1])[0]
        self.class_samples[dominant_class].append((image_path, mask_path))
    
    def get_test_sample(self) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Get a random test sample with ground truth masks."""
        idx = random.randint(0, len(self.images) - 1)
        image = self.load_image(self.images[idx])
        mask = self.load_mask(self.masks[idx])
        
        # Convert single mask to multi-class dictionary
        ground_truth = {}
        for i, class_name in enumerate(self.classes):
            class_mask = (mask == (i + 1)).float()  # +1 because 0 is background
            ground_truth[class_name] = class_mask
        
        return image, ground_truth 