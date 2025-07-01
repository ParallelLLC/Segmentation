"""
SAM 2 Zero-Shot Segmentation Model

This module implements zero-shot segmentation using SAM 2 with advanced
text prompting, visual grounding, and attention-based prompt generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import clip
from segment_anything_2 import sam_model_registry, SamPredictor
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel
import cv2


class SAM2ZeroShot(nn.Module):
    """
    SAM 2 Zero-Shot Segmentation Model
    
    Performs zero-shot segmentation using SAM 2 with advanced text prompting
    and visual grounding techniques.
    """
    
    def __init__(
        self,
        sam2_checkpoint: str,
        clip_model_name: str = "ViT-B/32",
        device: str = "cuda",
        use_attention_maps: bool = True,
        use_grounding_dino: bool = False,
        temperature: float = 0.1
    ):
        super().__init__()
        self.device = device
        self.temperature = temperature
        self.use_attention_maps = use_attention_maps
        self.use_grounding_dino = use_grounding_dino
        
        # Initialize SAM 2
        self.sam2 = sam_model_registry["vit_h"](checkpoint=sam2_checkpoint)
        self.sam2.to(device)
        self.sam2_predictor = SamPredictor(self.sam2)
        
        # Initialize CLIP
        self.clip_model, self.clip_preprocess = clip.load(clip_model_name, device=device)
        self.clip_model.eval()
        
        # Initialize CLIP text and vision models for attention
        if self.use_attention_maps:
            self.clip_text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_vision_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_text_model.to(device)
            self.clip_vision_model.to(device)
        
        # Advanced prompt templates with domain-specific variations
        self.advanced_prompts = {
            "satellite": {
                "building": [
                    "satellite view of buildings", "aerial photograph of structures",
                    "overhead view of houses", "urban development from above",
                    "rooftop structures", "architectural features from space"
                ],
                "road": [
                    "satellite view of roads", "aerial photograph of streets",
                    "overhead view of highways", "transportation network from above",
                    "paved surfaces", "road infrastructure from space"
                ],
                "vegetation": [
                    "satellite view of vegetation", "aerial photograph of forests",
                    "overhead view of trees", "green areas from above",
                    "natural landscape", "plant life from space"
                ],
                "water": [
                    "satellite view of water", "aerial photograph of lakes",
                    "overhead view of rivers", "water bodies from above",
                    "aquatic features", "water resources from space"
                ]
            },
            "fashion": {
                "shirt": [
                    "fashion photography of shirts", "clothing item top",
                    "apparel garment", "upper body clothing",
                    "casual wear", "formal attire top"
                ],
                "pants": [
                    "fashion photography of pants", "lower body clothing",
                    "trousers garment", "leg wear",
                    "casual pants", "formal trousers"
                ],
                "dress": [
                    "fashion photography of dresses", "full body garment",
                    "formal dress", "evening wear",
                    "casual dress", "party dress"
                ],
                "shoes": [
                    "fashion photography of shoes", "footwear item",
                    "foot covering", "walking shoes",
                    "casual footwear", "formal shoes"
                ]
            },
            "robotics": {
                "robot": [
                    "robotics environment with robot", "automation equipment",
                    "mechanical arm", "industrial robot",
                    "automated system", "robotic device"
                ],
                "tool": [
                    "robotics environment with tools", "industrial equipment",
                    "mechanical tools", "work equipment",
                    "hand tools", "power tools"
                ],
                "safety": [
                    "robotics environment with safety equipment", "protective gear",
                    "safety helmet", "safety vest",
                    "protective clothing", "safety equipment"
                ]
            }
        }
        
        # Prompt enhancement strategies
        self.prompt_strategies = {
            "descriptive": lambda x: f"a clear image showing {x}",
            "contextual": lambda x: f"in a typical environment, {x}",
            "detailed": lambda x: f"high quality photograph of {x} with clear details",
            "contrastive": lambda x: f"{x} standing out from the background"
        }
    
    def generate_attention_maps(
        self, 
        image: torch.Tensor, 
        text_prompts: List[str]
    ) -> torch.Tensor:
        """Generate attention maps using CLIP's cross-attention."""
        if not self.use_attention_maps:
            return None
        
        # Tokenize text prompts
        text_inputs = self.clip_tokenizer(
            text_prompts, 
            padding=True, 
            return_tensors="pt"
        ).to(self.device)
        
        # Get image features
        image_inputs = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        
        # Get attention maps from CLIP
        with torch.no_grad():
            text_outputs = self.clip_text_model(**text_inputs, output_attentions=True)
            vision_outputs = self.clip_vision_model(image_inputs, output_attentions=True)
            
            # Extract cross-attention maps
            cross_attention = text_outputs.cross_attentions[-1]  # Last layer
            attention_maps = cross_attention.mean(dim=1)  # Average over heads
        
        return attention_maps
    
    def extract_attention_points(
        self, 
        attention_maps: torch.Tensor, 
        num_points: int = 5
    ) -> List[Tuple[int, int]]:
        """Extract points from attention maps for SAM 2 prompting."""
        if attention_maps is None:
            return []
        
        # Resize attention map to image size
        h, w = attention_maps.shape[-2:]
        attention_maps = F.interpolate(
            attention_maps.unsqueeze(0), 
            size=(h, w), 
            mode='bilinear'
        ).squeeze(0)
        
        # Find top attention points
        points = []
        for i in range(min(num_points, attention_maps.shape[0])):
            attention_map = attention_maps[i]
            max_idx = torch.argmax(attention_map)
            y, x = max_idx // w, max_idx % w
            points.append((int(x), int(y)))
        
        return points
    
    def generate_enhanced_prompts(
        self, 
        domain: str, 
        class_names: List[str]
    ) -> List[str]:
        """Generate enhanced prompts using multiple strategies."""
        enhanced_prompts = []
        
        for class_name in class_names:
            if domain in self.advanced_prompts and class_name in self.advanced_prompts[domain]:
                base_prompts = self.advanced_prompts[domain][class_name]
                
                # Add base prompts
                enhanced_prompts.extend(base_prompts)
                
                # Add strategy-enhanced prompts
                for strategy_name, strategy_func in self.prompt_strategies.items():
                    for base_prompt in base_prompts[:2]:  # Use first 2 base prompts
                        enhanced_prompt = strategy_func(base_prompt)
                        enhanced_prompts.append(enhanced_prompt)
            else:
                # Fallback for unknown classes
                enhanced_prompts.append(class_name)
                enhanced_prompts.append(f"object: {class_name}")
        
        return enhanced_prompts
    
    def compute_text_image_similarity(
        self, 
        image: torch.Tensor, 
        text_prompts: List[str]
    ) -> torch.Tensor:
        """Compute similarity between image and text prompts."""
        # Tokenize and encode text
        text_tokens = clip.tokenize(text_prompts).to(self.device)
        
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_tokens)
            text_features = F.normalize(text_features, dim=-1)
            
            # Encode image
            image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            image_features = self.clip_model.encode_image(image_input)
            image_features = F.normalize(image_features, dim=-1)
            
            # Compute similarity
            similarity = torch.matmul(image_features, text_features.T) / self.temperature
        
        return similarity
    
    def generate_sam2_prompts(
        self, 
        image: torch.Tensor, 
        domain: str, 
        class_names: List[str]
    ) -> List[Dict]:
        """Generate comprehensive SAM 2 prompts for zero-shot segmentation."""
        prompts = []
        
        # Generate enhanced text prompts
        text_prompts = self.generate_enhanced_prompts(domain, class_names)
        
        # Compute text-image similarity
        similarities = self.compute_text_image_similarity(image, text_prompts)
        
        # Generate attention maps
        attention_maps = self.generate_attention_maps(image, text_prompts)
        attention_points = self.extract_attention_points(attention_maps)
        
        # Create prompts for each class
        for i, class_name in enumerate(class_names):
            class_prompts = []
            
            # Find relevant text prompts for this class
            class_text_indices = []
            for j, prompt in enumerate(text_prompts):
                if class_name.lower() in prompt.lower():
                    class_text_indices.append(j)
            
            if class_text_indices:
                # Get best similarity for this class
                class_similarities = similarities[0, class_text_indices]
                best_idx = torch.argmax(class_similarities)
                best_similarity = class_similarities[best_idx]
                
                if best_similarity > 0.2:  # Threshold for relevance
                    # Add attention-based points
                    if attention_points:
                        for point in attention_points[:3]:  # Use top 3 points
                            prompts.append({
                                'type': 'point',
                                'data': point,
                                'label': 1,
                                'class': class_name,
                                'confidence': best_similarity.item(),
                                'source': 'attention'
                            })
                    
                    # Add center point as fallback
                    h, w = image.shape[-2:]
                    center_point = [w // 2, h // 2]
                    prompts.append({
                        'type': 'point',
                        'data': center_point,
                        'label': 1,
                        'class': class_name,
                        'confidence': best_similarity.item(),
                        'source': 'center'
                    })
                    
                    # Add bounding box prompt (simple rectangle)
                    if best_similarity > 0.4:  # Higher threshold for box prompts
                        box = [w // 4, h // 4, 3 * w // 4, 3 * h // 4]
                        prompts.append({
                            'type': 'box',
                            'data': box,
                            'class': class_name,
                            'confidence': best_similarity.item(),
                            'source': 'similarity'
                        })
        
        return prompts
    
    def segment(
        self, 
        image: torch.Tensor, 
        domain: str, 
        class_names: List[str]
    ) -> Dict[str, torch.Tensor]:
        """
        Perform zero-shot segmentation.
        
        Args:
            image: Input image tensor [C, H, W]
            domain: Domain name (satellite, fashion, robotics)
            class_names: List of class names to segment
            
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
        prompts = self.generate_sam2_prompts(image, domain, class_names)
        
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
                if prompt['confidence'] > 0.2:
                    if class_name not in results:
                        results[class_name] = mask
                    else:
                        # Combine masks if multiple prompts for same class
                        results[class_name] = torch.max(results[class_name], mask)
            
            elif prompt['type'] == 'box':
                box = prompt['data']
                
                # Get SAM 2 prediction with box
                masks, scores, logits = self.sam2_predictor.predict(
                    box=np.array(box),
                    multimask_output=True
                )
                
                # Select best mask
                best_mask_idx = np.argmax(scores)
                mask = torch.from_numpy(masks[best_mask_idx]).float()
                
                # Apply confidence threshold
                if prompt['confidence'] > 0.3:
                    if class_name not in results:
                        results[class_name] = mask
                    else:
                        # Combine masks
                        results[class_name] = torch.max(results[class_name], mask)
        
        return results
    
    def forward(
        self, 
        image: torch.Tensor, 
        domain: str, 
        class_names: List[str]
    ) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        return self.segment(image, domain, class_names)


class ZeroShotEvaluator:
    """Evaluator for zero-shot segmentation."""
    
    def __init__(self):
        self.metrics = {}
    
    def compute_iou(self, pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> float:
        """Compute Intersection over Union."""
        intersection = (pred_mask & gt_mask).sum()
        union = (pred_mask | gt_mask).sum()
        return (intersection / union).item() if union > 0 else 0.0
    
    def compute_dice(self, pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> float:
        """Compute Dice coefficient."""
        intersection = (pred_mask & gt_mask).sum()
        total = pred_mask.sum() + gt_mask.sum()
        return (2 * intersection / total).item() if total > 0 else 0.0
    
    def evaluate(
        self, 
        predictions: Dict[str, torch.Tensor], 
        ground_truth: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Evaluate zero-shot segmentation results."""
        results = {}
        
        for class_name in ground_truth.keys():
            if class_name in predictions:
                pred_mask = predictions[class_name] > 0.5  # Threshold
                gt_mask = ground_truth[class_name] > 0.5
                
                iou = self.compute_iou(pred_mask, gt_mask)
                dice = self.compute_dice(pred_mask, gt_mask)
                
                results[f"{class_name}_iou"] = iou
                results[f"{class_name}_dice"] = dice
        
        # Compute average metrics
        if results:
            results['mean_iou'] = np.mean([v for k, v in results.items() if 'iou' in k])
            results['mean_dice'] = np.mean([v for k, v in results.items() if 'dice' in k])
        
        return results 