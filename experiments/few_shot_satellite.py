"""
Few-Shot Satellite Imagery Segmentation Experiment

This experiment demonstrates few-shot learning for satellite imagery segmentation
using SAM 2 with minimal labeled examples.
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

from models.sam2_fewshot import SAM2FewShot, FewShotTrainer
from utils.data_loader import SatelliteDataLoader
from utils.metrics import SegmentationMetrics
from utils.visualization import visualize_segmentation


class SatelliteFewShotExperiment:
    """Few-shot learning experiment for satellite imagery."""
    
    def __init__(
        self,
        sam2_checkpoint: str,
        data_dir: str,
        output_dir: str,
        device: str = "cuda",
        num_shots: int = 5,
        num_classes: int = 4
    ):
        self.device = device
        self.num_shots = num_shots
        self.num_classes = num_classes
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize model
        self.model = SAM2FewShot(
            sam2_checkpoint=sam2_checkpoint,
            device=device,
            prompt_engineering=True,
            visual_similarity=True
        )
        
        # Initialize trainer
        self.trainer = FewShotTrainer(self.model, learning_rate=1e-4)
        
        # Initialize data loader
        self.data_loader = SatelliteDataLoader(data_dir)
        
        # Initialize metrics
        self.metrics = SegmentationMetrics()
        
        # Satellite-specific classes
        self.classes = ["building", "road", "vegetation", "water"]
        
    def load_support_examples(self, class_name: str) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Load support examples for a specific class."""
        support_images, support_masks = [], []
        
        # Load few examples for this class
        examples = self.data_loader.get_class_examples(class_name, self.num_shots)
        
        for example in examples:
            image, mask = example
            support_images.append(image)
            support_masks.append(mask)
        
        return support_images, support_masks
    
    def run_episode(
        self, 
        query_image: torch.Tensor, 
        query_mask: torch.Tensor, 
        class_name: str
    ) -> Dict:
        """Run a single few-shot episode."""
        # Load support examples
        support_images, support_masks = self.load_support_examples(class_name)
        
        # Add support examples to model memory
        for img, mask in zip(support_images, support_masks):
            self.model.add_few_shot_example("satellite", class_name, img, mask)
        
        # Perform segmentation
        predictions = self.model.segment(
            query_image, 
            "satellite", 
            [class_name], 
            use_few_shot=True
        )
        
        # Compute metrics
        if class_name in predictions:
            pred_mask = predictions[class_name]
            metrics = self.metrics.compute_metrics(pred_mask, query_mask)
        else:
            metrics = {
                'iou': 0.0,
                'dice': 0.0,
                'precision': 0.0,
                'recall': 0.0
            }
        
        return {
            'predictions': predictions,
            'metrics': metrics,
            'support_images': support_images,
            'support_masks': support_masks
        }
    
    def run_experiment(self, num_episodes: int = 100) -> Dict:
        """Run the complete few-shot experiment."""
        results = {
            'episodes': [],
            'class_metrics': {cls: [] for cls in self.classes},
            'overall_metrics': []
        }
        
        print(f"Running {num_episodes} few-shot episodes...")
        
        for episode in tqdm(range(num_episodes)):
            # Sample random class and query image
            class_name = np.random.choice(self.classes)
            query_image, query_mask = self.data_loader.get_random_query(class_name)
            
            # Run episode
            episode_result = self.run_episode(query_image, query_mask, class_name)
            
            # Store results
            results['episodes'].append({
                'episode': episode,
                'class': class_name,
                'metrics': episode_result['metrics']
            })
            
            results['class_metrics'][class_name].append(episode_result['metrics'])
            
            # Compute overall metrics
            overall_metrics = {
                'mean_iou': np.mean([ep['metrics']['iou'] for ep in results['episodes']]),
                'mean_dice': np.mean([ep['metrics']['dice'] for ep in results['episodes']]),
                'mean_precision': np.mean([ep['metrics']['precision'] for ep in results['episodes']]),
                'mean_recall': np.mean([ep['metrics']['recall'] for ep in results['episodes']])
            }
            results['overall_metrics'].append(overall_metrics)
            
            # Visualize every 20 episodes
            if episode % 20 == 0:
                self.visualize_episode(
                    episode, 
                    query_image, 
                    query_mask, 
                    episode_result['predictions'],
                    episode_result['support_images'],
                    episode_result['support_masks'],
                    class_name
                )
        
        return results
    
    def visualize_episode(
        self,
        episode: int,
        query_image: torch.Tensor,
        query_mask: torch.Tensor,
        predictions: Dict[str, torch.Tensor],
        support_images: List[torch.Tensor],
        support_masks: List[torch.Tensor],
        class_name: str
    ):
        """Visualize a few-shot episode."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Query image
        axes[0, 0].imshow(query_image.permute(1, 2, 0).cpu().numpy())
        axes[0, 0].set_title(f"Query Image - {class_name}")
        axes[0, 0].axis('off')
        
        # Ground truth
        axes[0, 1].imshow(query_mask.cpu().numpy(), cmap='gray')
        axes[0, 1].set_title("Ground Truth")
        axes[0, 1].axis('off')
        
        # Prediction
        if class_name in predictions:
            pred_mask = predictions[class_name]
            axes[0, 2].imshow(pred_mask.cpu().numpy(), cmap='gray')
            axes[0, 2].set_title("Prediction")
        else:
            axes[0, 2].text(0.5, 0.5, "No Prediction", ha='center', va='center')
        axes[0, 2].axis('off')
        
        # Support examples
        for i in range(min(3, len(support_images))):
            axes[1, i].imshow(support_images[i].permute(1, 2, 0).cpu().numpy())
            axes[1, i].set_title(f"Support {i+1}")
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"episode_{episode}.png"))
        plt.close()
    
    def save_results(self, results: Dict):
        """Save experiment results."""
        # Save metrics
        with open(os.path.join(self.output_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save summary
        summary = {
            'num_episodes': len(results['episodes']),
            'num_shots': self.num_shots,
            'classes': self.classes,
            'final_metrics': results['overall_metrics'][-1] if results['overall_metrics'] else {},
            'class_averages': {}
        }
        
        for class_name in self.classes:
            if results['class_metrics'][class_name]:
                class_metrics = results['class_metrics'][class_name]
                summary['class_averages'][class_name] = {
                    'mean_iou': np.mean([m['iou'] for m in class_metrics]),
                    'mean_dice': np.mean([m['dice'] for m in class_metrics]),
                    'std_iou': np.std([m['iou'] for m in class_metrics]),
                    'std_dice': np.std([m['dice'] for m in class_metrics])
                }
        
        with open(os.path.join(self.output_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Results saved to {self.output_dir}")
        print(f"Final mean IoU: {summary['final_metrics'].get('mean_iou', 0):.3f}")
        print(f"Final mean Dice: {summary['final_metrics'].get('mean_dice', 0):.3f}")


def main():
    parser = argparse.ArgumentParser(description="Few-shot satellite segmentation experiment")
    parser.add_argument("--sam2_checkpoint", type=str, required=True, help="Path to SAM 2 checkpoint")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to satellite dataset")
    parser.add_argument("--output_dir", type=str, default="results/few_shot_satellite", help="Output directory")
    parser.add_argument("--num_shots", type=int, default=5, help="Number of support examples")
    parser.add_argument("--num_episodes", type=int, default=100, help="Number of episodes")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    # Run experiment
    experiment = SatelliteFewShotExperiment(
        sam2_checkpoint=args.sam2_checkpoint,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=args.device,
        num_shots=args.num_shots
    )
    
    results = experiment.run_experiment(num_episodes=args.num_episodes)
    experiment.save_results(results)


if __name__ == "__main__":
    main() 