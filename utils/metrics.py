"""
Segmentation Metrics

This module provides comprehensive metrics for evaluating segmentation performance
in few-shot and zero-shot learning scenarios.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import precision_recall_curve, average_precision_score
import cv2


class SegmentationMetrics:
    """Comprehensive segmentation metrics calculator."""
    
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
    
    def compute_metrics(
        self, 
        pred_mask: torch.Tensor, 
        gt_mask: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute comprehensive segmentation metrics.
        
        Args:
            pred_mask: Predicted mask tensor [H, W] or [1, H, W]
            gt_mask: Ground truth mask tensor [H, W] or [1, H, W]
            
        Returns:
            Dictionary containing various metrics
        """
        # Ensure masks are 2D
        if pred_mask.dim() == 3:
            pred_mask = pred_mask.squeeze(0)
        if gt_mask.dim() == 3:
            gt_mask = gt_mask.squeeze(0)
        
        # Convert to binary masks
        pred_binary = (pred_mask > self.threshold).float()
        gt_binary = (gt_mask > self.threshold).float()
        
        # Compute basic metrics
        metrics = {}
        
        # IoU (Intersection over Union)
        metrics['iou'] = self.compute_iou(pred_binary, gt_binary)
        
        # Dice coefficient
        metrics['dice'] = self.compute_dice(pred_binary, gt_binary)
        
        # Precision and Recall
        metrics['precision'] = self.compute_precision(pred_binary, gt_binary)
        metrics['recall'] = self.compute_recall(pred_binary, gt_binary)
        
        # F1 Score
        metrics['f1'] = self.compute_f1_score(pred_binary, gt_binary)
        
        # Accuracy
        metrics['accuracy'] = self.compute_accuracy(pred_binary, gt_binary)
        
        # Boundary metrics
        metrics['boundary_iou'] = self.compute_boundary_iou(pred_binary, gt_binary)
        metrics['hausdorff_distance'] = self.compute_hausdorff_distance(pred_binary, gt_binary)
        
        # Area metrics
        metrics['area_ratio'] = self.compute_area_ratio(pred_binary, gt_binary)
        
        return metrics
    
    def compute_iou(self, pred: torch.Tensor, gt: torch.Tensor) -> float:
        """Compute Intersection over Union."""
        intersection = (pred & gt).sum()
        union = (pred | gt).sum()
        return (intersection / union).item() if union > 0 else 0.0
    
    def compute_dice(self, pred: torch.Tensor, gt: torch.Tensor) -> float:
        """Compute Dice coefficient."""
        intersection = (pred & gt).sum()
        total = pred.sum() + gt.sum()
        return (2 * intersection / total).item() if total > 0 else 0.0
    
    def compute_precision(self, pred: torch.Tensor, gt: torch.Tensor) -> float:
        """Compute precision."""
        intersection = (pred & gt).sum()
        return (intersection / pred.sum()).item() if pred.sum() > 0 else 0.0
    
    def compute_recall(self, pred: torch.Tensor, gt: torch.Tensor) -> float:
        """Compute recall."""
        intersection = (pred & gt).sum()
        return (intersection / gt.sum()).item() if gt.sum() > 0 else 0.0
    
    def compute_f1_score(self, pred: torch.Tensor, gt: torch.Tensor) -> float:
        """Compute F1 score."""
        precision = self.compute_precision(pred, gt)
        recall = self.compute_recall(pred, gt)
        return (2 * precision * recall / (precision + recall)).item() if (precision + recall) > 0 else 0.0
    
    def compute_accuracy(self, pred: torch.Tensor, gt: torch.Tensor) -> float:
        """Compute pixel accuracy."""
        correct = (pred == gt).sum()
        total = pred.numel()
        return (correct / total).item()
    
    def compute_boundary_iou(self, pred: torch.Tensor, gt: torch.Tensor) -> float:
        """Compute boundary IoU."""
        # Extract boundaries
        pred_boundary = self.extract_boundary(pred)
        gt_boundary = self.extract_boundary(gt)
        
        # Compute IoU on boundaries
        return self.compute_iou(pred_boundary, gt_boundary)
    
    def extract_boundary(self, mask: torch.Tensor) -> torch.Tensor:
        """Extract boundary from binary mask."""
        mask_np = mask.cpu().numpy().astype(np.uint8)
        
        # Use morphological operations to extract boundary
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(mask_np, kernel, iterations=1)
        eroded = cv2.erode(mask_np, kernel, iterations=1)
        boundary = dilated - eroded
        
        return torch.from_numpy(boundary).float()
    
    def compute_hausdorff_distance(self, pred: torch.Tensor, gt: torch.Tensor) -> float:
        """Compute Hausdorff distance between boundaries."""
        pred_boundary = self.extract_boundary(pred)
        gt_boundary = self.extract_boundary(gt)
        
        # Convert to numpy for distance computation
        pred_np = pred_boundary.cpu().numpy()
        gt_np = gt_boundary.cpu().numpy()
        
        # Find boundary points
        pred_points = np.column_stack(np.where(pred_np > 0))
        gt_points = np.column_stack(np.where(gt_np > 0))
        
        if len(pred_points) == 0 or len(gt_points) == 0:
            return float('inf')
        
        # Compute Hausdorff distance
        hausdorff_dist = self._hausdorff_distance(pred_points, gt_points)
        return hausdorff_dist
    
    def _hausdorff_distance(self, set1: np.ndarray, set2: np.ndarray) -> float:
        """Compute Hausdorff distance between two point sets."""
        def directed_hausdorff(set_a, set_b):
            min_distances = []
            for point_a in set_a:
                distances = np.linalg.norm(set_b - point_a, axis=1)
                min_distances.append(np.min(distances))
            return np.max(min_distances)
        
        d1 = directed_hausdorff(set1, set2)
        d2 = directed_hausdorff(set2, set1)
        return max(d1, d2)
    
    def compute_area_ratio(self, pred: torch.Tensor, gt: torch.Tensor) -> float:
        """Compute ratio of predicted area to ground truth area."""
        pred_area = pred.sum()
        gt_area = gt.sum()
        return (pred_area / gt_area).item() if gt_area > 0 else 0.0
    
    def compute_class_metrics(
        self, 
        predictions: Dict[str, torch.Tensor], 
        ground_truth: Dict[str, torch.Tensor]
    ) -> Dict[str, Dict[str, float]]:
        """Compute metrics for multiple classes."""
        class_metrics = {}
        
        for class_name in ground_truth.keys():
            if class_name in predictions:
                metrics = self.compute_metrics(predictions[class_name], ground_truth[class_name])
                class_metrics[class_name] = metrics
            else:
                # No prediction for this class
                class_metrics[class_name] = {
                    'iou': 0.0,
                    'dice': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1': 0.0,
                    'accuracy': 0.0,
                    'boundary_iou': 0.0,
                    'hausdorff_distance': float('inf'),
                    'area_ratio': 0.0
                }
        
        return class_metrics
    
    def compute_average_metrics(
        self, 
        class_metrics: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """Compute average metrics across all classes."""
        if not class_metrics:
            return {}
        
        # Collect all metric names
        metric_names = list(class_metrics[list(class_metrics.keys())[0]].keys())
        
        # Compute averages
        averages = {}
        for metric_name in metric_names:
            values = [class_metrics[cls][metric_name] for cls in class_metrics.keys()]
            
            # Handle infinite values in Hausdorff distance
            if metric_name == 'hausdorff_distance':
                finite_values = [v for v in values if v != float('inf')]
                if finite_values:
                    averages[metric_name] = np.mean(finite_values)
                else:
                    averages[metric_name] = float('inf')
            else:
                averages[metric_name] = np.mean(values)
        
        return averages


class FewShotMetrics:
    """Specialized metrics for few-shot learning evaluation."""
    
    def __init__(self):
        self.segmentation_metrics = SegmentationMetrics()
    
    def compute_episode_metrics(
        self, 
        episode_results: List[Dict]
    ) -> Dict[str, float]:
        """Compute metrics across multiple episodes."""
        all_metrics = []
        
        for episode in episode_results:
            if 'metrics' in episode:
                all_metrics.append(episode['metrics'])
        
        if not all_metrics:
            return {}
        
        # Compute episode-level statistics
        episode_stats = {}
        metric_names = all_metrics[0].keys()
        
        for metric_name in metric_names:
            values = [ep[metric_name] for ep in all_metrics if metric_name in ep]
            if values:
                episode_stats[f'mean_{metric_name}'] = np.mean(values)
                episode_stats[f'std_{metric_name}'] = np.std(values)
                episode_stats[f'min_{metric_name}'] = np.min(values)
                episode_stats[f'max_{metric_name}'] = np.max(values)
        
        return episode_stats
    
    def compute_shot_analysis(
        self, 
        results_by_shots: Dict[int, List[Dict]]
    ) -> Dict[str, Dict[str, float]]:
        """Analyze performance across different numbers of shots."""
        shot_analysis = {}
        
        for num_shots, results in results_by_shots.items():
            episode_metrics = self.compute_episode_metrics(results)
            shot_analysis[f'{num_shots}_shots'] = episode_metrics
        
        return shot_analysis


class ZeroShotMetrics:
    """Specialized metrics for zero-shot learning evaluation."""
    
    def __init__(self):
        self.segmentation_metrics = SegmentationMetrics()
    
    def compute_prompt_strategy_comparison(
        self, 
        strategy_results: Dict[str, List[Dict]]
    ) -> Dict[str, Dict[str, float]]:
        """Compare different prompt strategies."""
        strategy_comparison = {}
        
        for strategy_name, results in strategy_results.items():
            # Compute average metrics for this strategy
            avg_metrics = {}
            if results:
                metric_names = results[0].keys()
                for metric_name in metric_names:
                    values = [r[metric_name] for r in results if metric_name in r]
                    if values:
                        avg_metrics[f'mean_{metric_name}'] = np.mean(values)
                        avg_metrics[f'std_{metric_name}'] = np.std(values)
            
            strategy_comparison[strategy_name] = avg_metrics
        
        return strategy_comparison
    
    def compute_attention_analysis(
        self, 
        with_attention: List[Dict], 
        without_attention: List[Dict]
    ) -> Dict[str, float]:
        """Analyze the impact of attention mechanisms."""
        if not with_attention or not without_attention:
            return {}
        
        # Compute average metrics
        with_attention_avg = {}
        without_attention_avg = {}
        
        metric_names = with_attention[0].keys()
        for metric_name in metric_names:
            with_values = [r[metric_name] for r in with_attention if metric_name in r]
            without_values = [r[metric_name] for r in without_attention if metric_name in r]
            
            if with_values:
                with_attention_avg[metric_name] = np.mean(with_values)
            if without_values:
                without_attention_avg[metric_name] = np.mean(without_values)
        
        # Compute improvements
        improvements = {}
        for metric_name in with_attention_avg.keys():
            if metric_name in without_attention_avg:
                improvement = with_attention_avg[metric_name] - without_attention_avg[metric_name]
                improvements[f'{metric_name}_improvement'] = improvement
        
        return {
            'with_attention': with_attention_avg,
            'without_attention': without_attention_avg,
            'improvements': improvements
        } 