"""
Visualization Utilities

This module provides comprehensive visualization tools for segmentation results,
attention maps, and experiment comparisons in few-shot and zero-shot learning.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import cv2
from PIL import Image
import os


class SegmentationVisualizer:
    """Visualization tools for segmentation results."""
    
    def __init__(self, figsize: Tuple[int, int] = (15, 10)):
        self.figsize = figsize
        
        # Color maps for different classes
        self.class_colors = {
            'building': [1.0, 0.0, 0.0],      # Red
            'road': [0.0, 1.0, 0.0],          # Green
            'vegetation': [0.0, 0.0, 1.0],    # Blue
            'water': [1.0, 1.0, 0.0],         # Yellow
            'shirt': [1.0, 0.5, 0.0],         # Orange
            'pants': [0.5, 0.0, 1.0],         # Purple
            'dress': [0.0, 1.0, 1.0],         # Cyan
            'shoes': [1.0, 0.0, 1.0],         # Magenta
            'robot': [0.5, 0.5, 0.5],         # Gray
            'tool': [0.8, 0.4, 0.2],          # Brown
            'safety': [0.2, 0.8, 0.2]         # Light Green
        }
    
    def visualize_segmentation(
        self, 
        image: torch.Tensor, 
        predictions: Dict[str, torch.Tensor], 
        ground_truth: Optional[Dict[str, torch.Tensor]] = None,
        title: str = "Segmentation Results"
    ) -> plt.Figure:
        """Visualize segmentation results with optional ground truth comparison."""
        num_classes = len(predictions)
        has_gt = ground_truth is not None
        
        # Calculate subplot layout
        if has_gt:
            cols = 3
            rows = max(2, num_classes)
        else:
            cols = 2
            rows = max(1, num_classes)
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        # Original image
        image_np = image.permute(1, 2, 0).cpu().numpy()
        # Denormalize if needed
        if image_np.min() < 0 or image_np.max() > 1:
            image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())
        
        axes[0, 0].imshow(image_np)
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis('off')
        
        # Combined prediction overlay
        if cols > 1:
            combined_pred = self.create_combined_mask(predictions)
            axes[0, 1].imshow(image_np)
            axes[0, 1].imshow(combined_pred, alpha=0.6, cmap='tab10')
            axes[0, 1].set_title("Combined Predictions")
            axes[0, 1].axis('off')
        
        # Ground truth overlay
        if has_gt and cols > 2:
            combined_gt = self.create_combined_mask(ground_truth)
            axes[0, 2].imshow(image_np)
            axes[0, 2].imshow(combined_gt, alpha=0.6, cmap='tab10')
            axes[0, 2].set_title("Ground Truth")
            axes[0, 2].axis('off')
        
        # Individual class predictions
        for i, (class_name, pred_mask) in enumerate(predictions.items()):
            row = i + 1 if has_gt else i
            col_offset = 0
            
            # Prediction mask
            pred_np = pred_mask.cpu().numpy()
            axes[row, col_offset].imshow(pred_np, cmap='gray')
            axes[row, col_offset].set_title(f"Prediction: {class_name}")
            axes[row, col_offset].axis('off')
            
            # Overlay on original image
            col_offset += 1
            axes[row, col_offset].imshow(image_np)
            axes[row, col_offset].imshow(pred_np, alpha=0.6, cmap='Reds')
            axes[row, col_offset].set_title(f"Overlay: {class_name}")
            axes[row, col_offset].axis('off')
            
            # Ground truth comparison
            if has_gt and class_name in ground_truth:
                col_offset += 1
                gt_mask = ground_truth[class_name]
                gt_np = gt_mask.cpu().numpy()
                
                # Create comparison visualization
                comparison = np.zeros((*gt_np.shape, 3))
                comparison[gt_np > 0.5] = [0, 1, 0]  # Green for ground truth
                comparison[pred_np > 0.5] = [1, 0, 0]  # Red for prediction
                comparison[(gt_np > 0.5) & (pred_np > 0.5)] = [1, 1, 0]  # Yellow for overlap
                
                axes[row, col_offset].imshow(image_np)
                axes[row, col_offset].imshow(comparison, alpha=0.6)
                axes[row, col_offset].set_title(f"Comparison: {class_name}")
                axes[row, col_offset].axis('off')
        
        plt.tight_layout()
        return fig
    
    def create_combined_mask(self, masks: Dict[str, torch.Tensor]) -> np.ndarray:
        """Create a combined mask visualization for multiple classes."""
        if not masks:
            return np.zeros((512, 512))
        
        # Get the shape from the first mask
        first_mask = list(masks.values())[0]
        combined = np.zeros((*first_mask.shape, 3))
        
        for i, (class_name, mask) in enumerate(masks.items()):
            mask_np = mask.cpu().numpy()
            color = self.class_colors.get(class_name, [1, 1, 1])
            
            # Apply color to mask
            for c in range(3):
                combined[:, :, c] += mask_np * color[c]
        
        # Normalize
        combined = np.clip(combined, 0, 1)
        return combined
    
    def visualize_attention_maps(
        self, 
        image: torch.Tensor, 
        attention_maps: torch.Tensor, 
        class_names: List[str],
        title: str = "Attention Maps"
    ) -> plt.Figure:
        """Visualize attention maps for different classes."""
        num_classes = len(class_names)
        fig, axes = plt.subplots(2, num_classes, figsize=(num_classes * 4, 8))
        
        # Original image
        image_np = image.permute(1, 2, 0).cpu().numpy()
        if image_np.min() < 0 or image_np.max() > 1:
            image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())
        
        for i in range(num_classes):
            axes[0, i].imshow(image_np)
            axes[0, i].set_title(f"Original - {class_names[i]}")
            axes[0, i].axis('off')
        
        # Attention maps
        attention_np = attention_maps.cpu().numpy()
        for i in range(min(num_classes, attention_np.shape[0])):
            attention_map = attention_np[i]
            
            # Resize attention map to image size
            attention_map = cv2.resize(attention_map, (image_np.shape[1], image_np.shape[0]))
            
            axes[1, i].imshow(attention_map, cmap='hot')
            axes[1, i].set_title(f"Attention - {class_names[i]}")
            axes[1, i].axis('off')
        
        plt.tight_layout()
        return fig
    
    def visualize_prompt_points(
        self, 
        image: torch.Tensor, 
        prompts: List[Dict],
        title: str = "Prompt Points"
    ) -> plt.Figure:
        """Visualize prompt points and boxes on the image."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        # Original image
        image_np = image.permute(1, 2, 0).cpu().numpy()
        if image_np.min() < 0 or image_np.max() > 1:
            image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())
        
        ax.imshow(image_np)
        
        # Plot prompts
        colors = plt.cm.Set3(np.linspace(0, 1, len(prompts)))
        
        for i, prompt in enumerate(prompts):
            color = colors[i]
            
            if prompt['type'] == 'point':
                x, y = prompt['data']
                ax.scatter(x, y, c=[color], s=100, marker='o', 
                          label=f"{prompt['class']} (point)")
                
            elif prompt['type'] == 'box':
                x1, y1, x2, y2 = prompt['data']
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                       linewidth=2, edgecolor=color, 
                                       facecolor='none', 
                                       label=f"{prompt['class']} (box)")
                ax.add_patch(rect)
        
        ax.set_title(title)
        ax.legend()
        ax.axis('off')
        
        return fig


class ExperimentVisualizer:
    """Visualization tools for experiment results and comparisons."""
    
    def __init__(self):
        self.segmentation_visualizer = SegmentationVisualizer()
    
    def plot_metrics_comparison(
        self, 
        results: Dict[str, List[float]], 
        metric_name: str = "IoU",
        title: str = "Metrics Comparison"
    ) -> plt.Figure:
        """Plot comparison of metrics across different methods/strategies."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Prepare data
        methods = list(results.keys())
        values = [np.mean(results[method]) for method in methods]
        errors = [np.std(results[method]) for method in methods]
        
        # Create bar plot
        bars = ax.bar(methods, values, yerr=errors, capsize=5, alpha=0.7)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
        
        ax.set_title(title)
        ax.set_ylabel(metric_name)
        ax.set_xlabel("Methods")
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig
    
    def plot_learning_curves(
        self, 
        episode_metrics: List[Dict[str, float]], 
        metric_name: str = "iou"
    ) -> plt.Figure:
        """Plot learning curves over episodes."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        # Extract metric values
        episodes = range(1, len(episode_metrics) + 1)
        values = [ep.get(metric_name, 0) for ep in episode_metrics]
        
        # Plot learning curve
        ax.plot(episodes, values, 'b-', linewidth=2, label=f'{metric_name.upper()}')
        
        # Add moving average
        window_size = min(10, len(values) // 4)
        if window_size > 1:
            moving_avg = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
            ax.plot(episodes[window_size-1:], moving_avg, 'r--', linewidth=2, 
                   label=f'Moving Average (window={window_size})')
        
        ax.set_title(f"Learning Curve - {metric_name.upper()}")
        ax.set_xlabel("Episode")
        ax.set_ylabel(metric_name.upper())
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    def plot_shot_analysis(
        self, 
        shot_results: Dict[int, List[float]], 
        metric_name: str = "iou"
    ) -> plt.Figure:
        """Plot performance analysis across different numbers of shots."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Prepare data
        shots = sorted(shot_results.keys())
        means = [np.mean(shot_results[shot]) for shot in shots]
        stds = [np.std(shot_results[shot]) for shot in shots]
        
        # Create line plot with error bars
        ax.errorbar(shots, means, yerr=stds, marker='o', linewidth=2, 
                   capsize=5, capthick=2)
        
        ax.set_title(f"Performance vs Number of Shots - {metric_name.upper()}")
        ax.set_xlabel("Number of Shots")
        ax.set_ylabel(f"Mean {metric_name.upper()}")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_prompt_strategy_comparison(
        self, 
        strategy_results: Dict[str, Dict[str, float]], 
        metric_name: str = "mean_iou"
    ) -> plt.Figure:
        """Plot comparison of different prompt strategies."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        # Prepare data
        strategies = list(strategy_results.keys())
        values = [strategy_results[s].get(metric_name, 0) for s in strategies]
        errors = [strategy_results[s].get(f'std_{metric_name.split("_")[-1]}', 0) 
                 for s in strategies]
        
        # Create bar plot
        bars = ax.bar(strategies, values, yerr=errors, capsize=5, alpha=0.7)
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
        
        ax.set_title(f"Prompt Strategy Comparison - {metric_name}")
        ax.set_ylabel(metric_name.replace('_', ' ').title())
        ax.set_xlabel("Strategy")
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig
    
    def create_comprehensive_report(
        self, 
        experiment_results: Dict,
        output_dir: str,
        experiment_name: str = "experiment"
    ):
        """Create a comprehensive visualization report."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Create summary plots
        if 'episode_metrics' in experiment_results:
            # Learning curves
            for metric in ['iou', 'dice', 'precision', 'recall']:
                fig = self.plot_learning_curves(
                    experiment_results['episode_metrics'], 
                    metric
                )
                fig.savefig(os.path.join(output_dir, f'{experiment_name}_learning_curve_{metric}.png'))
                plt.close(fig)
        
        if 'class_metrics' in experiment_results:
            # Class-wise performance
            class_results = experiment_results['class_metrics']
            for class_name, metrics in class_results.items():
                if isinstance(metrics, list):
                    fig = self.plot_learning_curves(metrics, 'iou')
                    fig.savefig(os.path.join(output_dir, f'{experiment_name}_class_{class_name}.png'))
                    plt.close(fig)
        
        if 'shot_analysis' in experiment_results:
            # Shot analysis
            for metric in ['iou', 'dice']:
                fig = self.plot_shot_analysis(
                    experiment_results['shot_analysis'], 
                    metric
                )
                fig.savefig(os.path.join(output_dir, f'{experiment_name}_shot_analysis_{metric}.png'))
                plt.close(fig)
        
        if 'strategy_comparison' in experiment_results:
            # Strategy comparison
            for metric in ['mean_iou', 'mean_dice']:
                fig = self.plot_prompt_strategy_comparison(
                    experiment_results['strategy_comparison'], 
                    metric
                )
                fig.savefig(os.path.join(output_dir, f'{experiment_name}_strategy_comparison_{metric}.png'))
                plt.close(fig)
        
        print(f"Comprehensive report saved to {output_dir}")


class AttentionVisualizer:
    """Specialized visualizer for attention mechanisms."""
    
    def __init__(self):
        self.segmentation_visualizer = SegmentationVisualizer()
    
    def visualize_cross_attention(
        self, 
        image: torch.Tensor, 
        text_tokens: List[str], 
        attention_weights: torch.Tensor,
        title: str = "Cross-Attention Visualization"
    ) -> plt.Figure:
        """Visualize cross-attention between image and text tokens."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Original image
        image_np = image.permute(1, 2, 0).cpu().numpy()
        if image_np.min() < 0 or image_np.max() > 1:
            image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())
        
        axes[0, 0].imshow(image_np)
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis('off')
        
        # Text tokens
        axes[0, 1].text(0.1, 0.5, '\n'.join(text_tokens), fontsize=12, 
                       verticalalignment='center')
        axes[0, 1].set_title("Text Tokens")
        axes[0, 1].axis('off')
        
        # Attention heatmap
        attention_np = attention_weights.cpu().numpy()
        sns.heatmap(attention_np, ax=axes[1, 0], cmap='viridis')
        axes[1, 0].set_title("Attention Heatmap")
        axes[1, 0].set_xlabel("Text Tokens")
        axes[1, 0].set_ylabel("Image Patches")
        
        # Attention overlay on image
        # Resize attention to image size
        attention_map = np.mean(attention_np, axis=1)
        attention_map = attention_map.reshape(int(np.sqrt(len(attention_map))), -1)
        attention_map = cv2.resize(attention_map, (image_np.shape[1], image_np.shape[0]))
        
        axes[1, 1].imshow(image_np)
        axes[1, 1].imshow(attention_map, alpha=0.6, cmap='hot')
        axes[1, 1].set_title("Attention Overlay")
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        return fig 