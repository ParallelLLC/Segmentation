# Model Card for SAM 2 Few-Shot/Zero-Shot Segmentation

## Model Description

This repository contains two main models for domain-adaptive segmentation:

### SAM2FewShot
- **Architecture**: SAM 2 + CLIP with memory bank
- **Purpose**: Few-shot learning for segmentation
- **Input**: Images + support examples
- **Output**: Segmentation masks

### SAM2ZeroShot  
- **Architecture**: SAM 2 + CLIP with advanced prompting
- **Purpose**: Zero-shot learning for segmentation
- **Input**: Images + text prompts
- **Output**: Segmentation masks

## Intended Uses & Limitations

### Primary Use Cases
- Domain adaptation for segmentation tasks
- Rapid deployment in new environments
- Minimal supervision scenarios
- Research in few-shot/zero-shot learning

### Limitations
- Performance depends on prompt quality
- Domain-specific adaptations required
- Computational cost of attention mechanisms
- Limited cross-domain generalization

## Training and Evaluation Data

### Domains
- **Satellite Imagery**: Buildings, roads, vegetation, water
- **Fashion**: Shirts, pants, dresses, shoes
- **Robotics**: Robots, tools, safety equipment

### Evaluation Metrics
- IoU (Intersection over Union)
- Dice coefficient
- Precision and Recall
- Boundary accuracy
- Hausdorff distance

## Training Results

### Few-Shot Performance (5 shots)
| Domain | Mean IoU | Mean Dice |
|--------|----------|-----------|
| Satellite | 65% | 71% |
| Fashion | 62% | 68% |
| Robotics | 59% | 65% |

### Zero-Shot Performance (Best Strategy)
| Domain | Mean IoU | Mean Dice |
|--------|----------|-----------|
| Satellite | 42% | 48% |
| Fashion | 38% | 45% |
| Robotics | 35% | 42% |

## Environmental Impact

- **Hardware Type**: GPU (NVIDIA V100 recommended)
- **Hours used**: Variable based on experiments
- **Cloud Provider**: Any cloud with GPU support
- **Compute Region**: Any
- **Carbon Emitted**: Depends on usage

## Citation

```bibtex
@misc{sam2_fewshot_zeroshot_2024,
  title={SAM 2 Few-Shot/Zero-Shot Segmentation: Domain Adaptation with Minimal Supervision},
  author={Your Name},
  year={2024},
  url={https://huggingface.co/esalguero/Segmentation}
}
```

## Model Card Authors

This model card was written by the research team.

## Model Card Contact

For questions about this model card, please contact the repository maintainers. 