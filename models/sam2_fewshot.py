"""
SAM 2 Few-Shot Learning Model

This module implements a few-shot segmentation model that combines SAM 2 with CLIP
for domain adaptation using minimal labeled examples.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import clip
from segment_anything_2 import sam_model_registry, SamPredictor
from transformers import CLIPTextModel, CLIPTokenizer


class SAM2FewShot(nn.Module):
    """
    SAM 2 Few-Shot Learning Model
    
    Combines SAM 2 with CLIP for few-shot and zero-shot segmentation
    across different domains (satellite, fashion, robotics).
    """
    
    def __init__(
        self,
        sam2_checkpoint: str,
        clip_model_name: str = "ViT-B/32",
        device: str = "cuda",
        prompt_engineering: bool = True,
        visual_similarity: bool = True,
        temperature: float = 0.1
    ):
        super().__init__()
        self.device = device
        self.temperature = temperature
        self.prompt_engineering = prompt_engineering
        self.visual_similarity = visual_similarity
        
        # Initialize SAM 2
        self.sam2 = sam_model_registry["vit_h"](checkpoint=sam2_checkpoint)
        self.sam2.to(device)
        self.sam2_predictor = SamPredictor(self.sam2)
        
        # Initialize CLIP for text and visual similarity
        self.clip_model, self.clip_preprocess = clip.load(clip_model_name, device=device)
        self.clip_model.eval()
        
        # Domain-specific prompt templates
        self.domain_prompts = {
            "satellite": {
                "building": ["building", "house", "structure", "rooftop"],
                "road": ["road", "street", "highway", "pavement"],
                "vegetation": ["vegetation", "forest", "trees", "green area"],
                "water": ["water", "lake", "river", "ocean", "pond"]
            },
            "fashion": {
                "shirt": ["shirt", "t-shirt", "blouse", "top"],
                "pants": ["pants", "trousers", "jeans", "legs"],
                "dress": ["dress", "gown", "outfit"],
                "shoes": ["shoes", "footwear", "sneakers", "boots"]
            },
            "robotics": {
                "robot": ["robot", "automation", "mechanical arm"],
                "tool": ["tool", "wrench", "screwdriver", "equipment"],
                "safety": ["safety equipment", "helmet", "vest", "protection"]
            }
        }
        
        # Few-shot memory bank
        self.few_shot_memory = {}
        
    def encode_text_prompts(self, domain: str, class_names: List[str]) -> torch.Tensor:
        """Encode text prompts for given domain and classes."""
        prompts = []
        for class_name in class_names:
            if domain in self.domain_prompts and class_name in self.domain_prompts[domain]:
                prompts.extend(self.domain_prompts[domain][class_name])
            else:
                prompts.append(class_name)
        
        # Add domain-specific context
        if domain == "satellite":
            prompts = [f"satellite image of {p}" for p in prompts]
        elif domain == "fashion":
            prompts = [f"fashion item {p}" for p in prompts]
        elif domain == "robotics":
            prompts = [f"robotics environment {p}" for p in prompts]
        
        text_tokens = clip.tokenize(prompts).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_tokens)
            text_features = F.normalize(text_features, dim=-1)
        
        return text_features
    
    def encode_image(self, image: Union[torch.Tensor, np.ndarray, Image.Image]) -> torch.Tensor:
        """Encode image using CLIP."""
        if isinstance(image, torch.Tensor):
            if image.dim() == 4:
                image = image.squeeze(0)
            image = image.permute(1, 2, 0).cpu().numpy()
        
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Preprocess for CLIP
        clip_image = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            image_features = self.clip_model.encode_image(clip_image)
            image_features = F.normalize(image_features, dim=-1)
        
        return image_features
    
    def compute_similarity(
        self, 
        image_features: torch.Tensor, 
        text_features: torch.Tensor
    ) -> torch.Tensor:
        """Compute similarity between image and text features."""
        similarity = torch.matmul(image_features, text_features.T) / self.temperature
        return similarity
    
    def add_few_shot_example(
        self, 
        domain: str, 
        class_name: str, 
        image: torch.Tensor, 
        mask: torch.Tensor
    ):
        """Add a few-shot example to the memory bank."""
        if domain not in self.few_shot_memory:
            self.few_shot_memory[domain] = {}
        
        if class_name not in self.few_shot_memory[domain]:
            self.few_shot_memory[domain][class_name] = []
        
        # Encode the example
        image_features = self.encode_image(image)
        
        self.few_shot_memory[domain][class_name].append({
            'image_features': image_features,
            'mask': mask,
            'image': image
        })
    
    def get_few_shot_similarity(
        self, 
        query_image: torch.Tensor, 
        domain: str, 
        class_name: str
    ) -> torch.Tensor:
        """Compute similarity with few-shot examples."""
        if domain not in self.few_shot_memory or class_name not in self.few_shot_memory[domain]:
            return torch.zeros(1, device=self.device)
        
        query_features = self.encode_image(query_image)
        similarities = []
        
        for example in self.few_shot_memory[domain][class_name]:
            similarity = F.cosine_similarity(
                query_features, 
                example['image_features'], 
                dim=-1
            )
            similarities.append(similarity)
        
        return torch.stack(similarities).mean()
    
    def generate_sam2_prompts(
        self, 
        image: torch.Tensor, 
        domain: str, 
        class_names: List[str],
        use_few_shot: bool = True
    ) -> List[Dict]:
        """Generate SAM 2 prompts based on text and few-shot similarity."""
        prompts = []
        
        # Text-based prompts
        if self.prompt_engineering:
            text_features = self.encode_text_prompts(domain, class_names)
            image_features = self.encode_image(image)
            text_similarities = self.compute_similarity(image_features, text_features)
            
            # Generate point prompts based on text similarity
            for i, class_name in enumerate(class_names):
                if text_similarities[0, i] > 0.3:  # Threshold for relevance
                    # Simple center point prompt (can be enhanced with attention maps)
                    h, w = image.shape[-2:]
                    point = [w // 2, h // 2]
                    prompts.append({
                        'type': 'point',
                        'data': point,
                        'label': 1,
                        'class': class_name,
                        'confidence': text_similarities[0, i].item()
                    })
        
        # Few-shot based prompts
        if use_few_shot and self.visual_similarity:
            for class_name in class_names:
                few_shot_sim = self.get_few_shot_similarity(image, domain, class_name)
                if few_shot_sim > 0.5:  # High similarity threshold
                    h, w = image.shape[-2:]
                    point = [w // 2, h // 2]
                    prompts.append({
                        'type': 'point',
                        'data': point,
                        'label': 1,
                        'class': class_name,
                        'confidence': few_shot_sim.item()
                    })
        
        return prompts
    
    def segment(
        self, 
        image: torch.Tensor, 
        domain: str, 
        class_names: List[str],
        use_few_shot: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Perform few-shot/zero-shot segmentation.
        
        Args:
            image: Input image tensor [C, H, W]
            domain: Domain name (satellite, fashion, robotics)
            class_names: List of class names to segment
            use_few_shot: Whether to use few-shot examples
            
        Returns:
            Dictionary with masks for each class
        """
        # Convert image for SAM 2
        if isinstance(image, torch.Tensor):
            image_np = image.permute(1, 2, 0).cpu().numpy()
        else:
            image_np = image
        
        # Set image in SAM 2 predictor
        self.sam2_predictor.set_image(image_np)
        
        # Generate prompts
        prompts = self.generate_sam2_prompts(image, domain, class_names, use_few_shot)
        
        results = {}
        
        for prompt in prompts:
            class_name = prompt['class']
            
            if prompt['type'] == 'point':
                point = prompt['data']
                label = prompt['label']
                
                # Get SAM 2 prediction
                masks, scores, logits = self.sam2_predictor.predict(
                    point_coords=np.array([point]),
                    point_labels=np.array([label]),
                    multimask_output=True
                )
                
                # Select best mask
                best_mask_idx = np.argmax(scores)
                mask = torch.from_numpy(masks[best_mask_idx]).float()
                
                # Apply confidence threshold
                if prompt['confidence'] > 0.3:
                    results[class_name] = mask
        
        return results
    
    def forward(
        self, 
        image: torch.Tensor, 
        domain: str, 
        class_names: List[str],
        use_few_shot: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for training."""
        return self.segment(image, domain, class_names, use_few_shot)


class FewShotTrainer:
    """Trainer for few-shot segmentation."""
    
    def __init__(self, model: SAM2FewShot, learning_rate: float = 1e-4):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.criterion = nn.BCELoss()
    
    def train_step(
        self, 
        support_images: List[torch.Tensor],
        support_masks: List[torch.Tensor],
        query_image: torch.Tensor,
        query_mask: torch.Tensor,
        domain: str,
        class_name: str
    ):
        """Single training step."""
        self.model.train()
        
        # Add support examples to memory
        for img, mask in zip(support_images, support_masks):
            self.model.add_few_shot_example(domain, class_name, img, mask)
        
        # Forward pass
        predictions = self.model(query_image, domain, [class_name], use_few_shot=True)
        
        if class_name in predictions:
            pred_mask = predictions[class_name]
            loss = self.criterion(pred_mask, query_mask)
        else:
            # If no prediction, use zero loss (can be improved)
            loss = torch.tensor(0.0, device=self.model.device, requires_grad=True)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item() 