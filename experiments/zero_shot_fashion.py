"""
Zero-Shot Fashion Segmentation Experiment

This experiment demonstrates zero-shot learning for fashion segmentation
using SAM 2 with advanced text prompting and attention mechanisms.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import json
from typing import List, Dict, Tuple
import argparse
from tqdm import tqdm

# Add parent directory to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.sam2_zeroshot import SAM2ZeroShot, ZeroShotEvaluator
from utils.data_loader import FashionDataLoader
from utils.metrics import SegmentationMetrics
from utils.visualization import visualize_segmentation


class FashionZeroShotExperiment:
    """Zero-shot learning experiment for fashion segmentation."""
    
    def __init__(
        self,
        sam2_checkpoint: str,
        data_dir: str,
        output_dir: str,
        device: str = "cuda",
        use_attention_maps: bool = True,
        temperature: float = 0.1
    ):
        self.device = device
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize model
        self.model = SAM2ZeroShot(
            sam2_checkpoint=sam2_checkpoint,
            device=device,
            use_attention_maps=use_attention_maps,
            temperature=temperature
        )
        
        # Initialize evaluator
        self.evaluator = ZeroShotEvaluator()
        
        # Initialize data loader
        self.data_loader = FashionDataLoader(data_dir)
        
        # Initialize metrics
        self.metrics = SegmentationMetrics()
        
        # Fashion-specific classes
        self.classes = ["shirt", "pants", "dress", "shoes"]
        
        # Prompt strategies to test
        self.prompt_strategies = [
            "basic",      # Simple class names
            "descriptive", # Enhanced descriptions
            "contextual",  # Context-aware prompts
            "detailed"     # Detailed descriptions
        ]
    
    def run_single_experiment(
        self, 
        image: torch.Tensor, 
        ground_truth: Dict[str, torch.Tensor],
        strategy: str = "descriptive"
    ) -> Dict:
        """Run a single zero-shot experiment."""
        # Perform segmentation
        predictions = self.model.segment(image, "fashion", self.classes)
        
        # Evaluate results
        evaluation = self.evaluator.evaluate(predictions, ground_truth)
        
        return {
            'predictions': predictions,
            'evaluation': evaluation,
            'strategy': strategy
        }
    
    def run_comparative_experiment(
        self, 
        num_images: int = 50
    ) -> Dict:
        """Run comparative experiment with different prompt strategies."""
        results = {
            'strategies': {strategy: [] for strategy in self.prompt_strategies},
            'overall_comparison': {},
            'class_analysis': {cls: {strategy: [] for strategy in self.prompt_strategies} 
                             for cls in self.classes}
        }
        
        print(f"Running comparative zero-shot experiment on {num_images} images...")
        
        for i in tqdm(range(num_images)):
            # Load test image and ground truth
            image, ground_truth = self.data_loader.get_test_sample()
            
            # Test each strategy
            for strategy in self.prompt_strategies:
                # Modify model's prompt strategy for this experiment
                if strategy == "basic":
                    # Use simple prompts
                    self.model.advanced_prompts["fashion"] = {
                        "shirt": ["shirt"],
                        "pants": ["pants"],
                        "dress": ["dress"],
                        "shoes": ["shoes"]
                    }
                elif strategy == "descriptive":
                    # Use descriptive prompts
                    self.model.advanced_prompts["fashion"] = {
                        "shirt": ["fashion photography of shirts", "clothing item top"],
                        "pants": ["fashion photography of pants", "lower body clothing"],
                        "dress": ["fashion photography of dresses", "full body garment"],
                        "shoes": ["fashion photography of shoes", "footwear item"]
                    }
                elif strategy == "contextual":
                    # Use contextual prompts
                    self.model.advanced_prompts["fashion"] = {
                        "shirt": ["in a fashion setting, shirt", "worn by a person, shirt"],
                        "pants": ["in a fashion setting, pants", "worn by a person, pants"],
                        "dress": ["in a fashion setting, dress", "worn by a person, dress"],
                        "shoes": ["in a fashion setting, shoes", "worn by a person, shoes"]
                    }
                elif strategy == "detailed":
                    # Use detailed prompts
                    self.model.advanced_prompts["fashion"] = {
                        "shirt": ["high quality fashion photograph of a shirt with clear details", 
                                "professional clothing photography showing shirt"],
                        "pants": ["high quality fashion photograph of pants with clear details",
                                "professional clothing photography showing pants"],
                        "dress": ["high quality fashion photograph of a dress with clear details",
                                "professional clothing photography showing dress"],
                        "shoes": ["high quality fashion photograph of shoes with clear details",
                                "professional clothing photography showing shoes"]
                    }
                
                # Run experiment
                experiment_result = self.run_single_experiment(image, ground_truth, strategy)
                
                # Store results
                results['strategies'][strategy].append(experiment_result['evaluation'])
                
                # Store class-specific results
                for class_name in self.classes:
                    iou_key = f"{class_name}_iou"
                    dice_key = f"{class_name}_dice"
                    
                    if iou_key in experiment_result['evaluation']:
                        results['class_analysis'][class_name][strategy].append({
                            'iou': experiment_result['evaluation'][iou_key],
                            'dice': experiment_result['evaluation'][dice_key]
                        })
                
                # Visualize every 10 images
                if i % 10 == 0:
                    self.visualize_comparison(
                        i, image, ground_truth, 
                        {s: results['strategies'][s][-1] for s in self.prompt_strategies},
                        strategy
                    )
        
        # Compute overall comparison
        for strategy in self.prompt_strategies:
            strategy_results = results['strategies'][strategy]
            if strategy_results:
                results['overall_comparison'][strategy] = {
                    'mean_iou': np.mean([r.get('mean_iou', 0) for r in strategy_results]),
                    'mean_dice': np.mean([r.get('mean_dice', 0) for r in strategy_results]),
                    'std_iou': np.std([r.get('mean_iou', 0) for r in strategy_results]),
                    'std_dice': np.std([r.get('mean_dice', 0) for r in strategy_results])
                }
        
        return results
    
    def run_attention_analysis(self, num_images: int = 20) -> Dict:
        """Run analysis of attention-based prompt generation."""
        results = {
            'with_attention': [],
            'without_attention': [],
            'attention_points': []
        }
        
        print(f"Running attention analysis on {num_images} images...")
        
        for i in tqdm(range(num_images)):
            # Load test image and ground truth
            image, ground_truth = self.data_loader.get_test_sample()
            
            # Test with attention maps
            self.model.use_attention_maps = True
            with_attention = self.run_single_experiment(image, ground_truth, "attention")
            
            # Test without attention maps
            self.model.use_attention_maps = False
            without_attention = self.run_single_experiment(image, ground_truth, "no_attention")
            
            # Store results
            results['with_attention'].append(with_attention['evaluation'])
            results['without_attention'].append(without_attention['evaluation'])
            
            # Analyze attention points
            if with_attention['predictions']:
                # Extract attention points (simplified)
                attention_points = self.extract_attention_points(image, self.classes)
                results['attention_points'].append(attention_points)
        
        return results
    
    def extract_attention_points(self, image: torch.Tensor, classes: List[str]) -> List[Tuple[int, int]]:
        """Extract attention points for visualization."""
        # Simplified attention point extraction
        h, w = image.shape[-2:]
        points = []
        
        for class_name in classes:
            # Generate some sample points (in practice, these would come from attention maps)
            center_x, center_y = w // 2, h // 2
            points.append((center_x, center_y))
            
            # Add some variation
            points.append((center_x + w // 4, center_y))
            points.append((center_x, center_y + h // 4))
        
        return points
    
    def visualize_comparison(
        self,
        image_idx: int,
        image: torch.Tensor,
        ground_truth: Dict[str, torch.Tensor],
        strategy_results: Dict,
        best_strategy: str
    ):
        """Visualize comparison between different strategies."""
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        
        # Original image
        axes[0, 0].imshow(image.permute(1, 2, 0).cpu().numpy())
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis('off')
        
        # Ground truth
        for i, class_name in enumerate(self.classes):
            if class_name in ground_truth:
                axes[0, i+1].imshow(ground_truth[class_name].cpu().numpy(), cmap='gray')
                axes[0, i+1].set_title(f"GT: {class_name}")
            axes[0, i+1].axis('off')
        
        # Best strategy predictions
        best_result = strategy_results[best_strategy]
        for i, class_name in enumerate(self.classes):
            if class_name in best_result:
                axes[1, i].imshow(best_result[class_name].cpu().numpy(), cmap='gray')
                axes[1, i].set_title(f"Best: {class_name}")
            axes[1, i].axis('off')
        
        # Strategy comparison
        strategies = list(strategy_results.keys())
        metrics = ['mean_iou', 'mean_dice']
        
        for i, metric in enumerate(metrics):
            values = [strategy_results[s].get(metric, 0) for s in strategies]
            axes[2, i].bar(strategies, values)
            axes[2, i].set_title(f"{metric.replace('_', ' ').title()}")
            axes[2, i].tick_params(axis='x', rotation=45)
        
        # Add text summary
        summary_text = f"Best Strategy: {best_strategy}\n"
        for strategy, result in strategy_results.items():
            summary_text += f"{strategy}: IoU={result.get('mean_iou', 0):.3f}, Dice={result.get('mean_dice', 0):.3f}\n"
        
        axes[2, 2].text(0.1, 0.5, summary_text, transform=axes[2, 2].transAxes, 
                       verticalalignment='center', fontsize=10)
        axes[2, 2].axis('off')
        axes[2, 3].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"comparison_{image_idx}.png"))
        plt.close()
    
    def save_results(self, results: Dict, experiment_type: str = "comparative"):
        """Save experiment results."""
        # Save detailed results
        with open(os.path.join(self.output_dir, f'{experiment_type}_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save summary
        if experiment_type == "comparative":
            summary = {
                'experiment_type': experiment_type,
                'num_images': len(results['strategies'][list(results['strategies'].keys())[0]]),
                'overall_comparison': results['overall_comparison'],
                'best_strategy': max(results['overall_comparison'].items(), 
                                   key=lambda x: x[1]['mean_iou'])[0]
            }
        else:
            summary = {
                'experiment_type': experiment_type,
                'attention_analysis': {
                    'with_attention_mean_iou': np.mean([r.get('mean_iou', 0) for r in results['with_attention']]),
                    'without_attention_mean_iou': np.mean([r.get('mean_iou', 0) for r in results['without_attention']]),
                    'attention_improvement': np.mean([r.get('mean_iou', 0) for r in results['with_attention']]) - 
                                          np.mean([r.get('mean_iou', 0) for r in results['without_attention']])
                }
            }
        
        with open(os.path.join(self.output_dir, f'{experiment_type}_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Results saved to {self.output_dir}")
        if experiment_type == "comparative":
            print(f"Best strategy: {summary['best_strategy']}")
            print(f"Best mean IoU: {summary['overall_comparison'][summary['best_strategy']]['mean_iou']:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Zero-shot fashion segmentation experiment")
    parser.add_argument("--sam2_checkpoint", type=str, required=True, help="Path to SAM 2 checkpoint")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to fashion dataset")
    parser.add_argument("--output_dir", type=str, default="results/zero_shot_fashion", help="Output directory")
    parser.add_argument("--num_images", type=int, default=50, help="Number of test images")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--experiment_type", type=str, default="comparative", 
                       choices=["comparative", "attention"], help="Type of experiment")
    parser.add_argument("--temperature", type=float, default=0.1, help="CLIP temperature")
    
    args = parser.parse_args()
    
    # Run experiment
    experiment = FashionZeroShotExperiment(
        sam2_checkpoint=args.sam2_checkpoint,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=args.device,
        temperature=args.temperature
    )
    
    if args.experiment_type == "comparative":
        results = experiment.run_comparative_experiment(num_images=args.num_images)
    else:
        results = experiment.run_attention_analysis(num_images=args.num_images)
    
    experiment.save_results(results, args.experiment_type)


if __name__ == "__main__":
    main() 