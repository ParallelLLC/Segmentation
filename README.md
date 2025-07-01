# SAM 2 Few-Shot/Zero-Shot Segmentation Research

This repository contains research on combining Segment Anything Model 2 (SAM 2) with minimal supervision for domain-specific segmentation tasks.

## Research Overview

The goal is to study how SAM 2 can be adapted to new object categories in specific domains (satellite imagery, fashion, robotics) using:
- **Few-shot learning**: 1-10 labeled examples per class
- **Zero-shot learning**: No labeled examples, using text prompts and visual similarity

## Key Research Areas

### 1. Domain Adaptation
- **Satellite Imagery**: Buildings, roads, vegetation, water bodies
- **Fashion**: Clothing items, accessories, patterns
- **Robotics**: Industrial objects, tools, safety equipment

### 2. Learning Paradigms
- **Prompt Engineering**: Optimizing text prompts for SAM 2
- **Visual Similarity**: Using CLIP embeddings for zero-shot transfer
- **Meta-learning**: Learning to adapt quickly to new domains

### 3. Evaluation Metrics
- IoU (Intersection over Union)
- Dice Coefficient
- Boundary Accuracy
- Domain-specific metrics

## Project Structure

```
├── data/                   # Dataset storage
├── models/                 # Model implementations
├── experiments/           # Experiment configurations
├── utils/                 # Utility functions
├── notebooks/             # Jupyter notebooks for analysis
├── results/               # Experiment results and visualizations
└── requirements.txt       # Dependencies
```

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Download SAM 2**:
   ```bash
   python scripts/download_sam2.py
   ```

3. **Run few-shot experiment**:
   ```bash
   python experiments/few_shot_satellite.py
   ```

4. **Run zero-shot experiment**:
   ```bash
   python experiments/zero_shot_fashion.py
   ```

## Research Papers

This work builds upon:
- [SAM 2: Segment Anything Model 2](https://arxiv.org/abs/2311.15796)
- [CLIP: Learning Transferable Visual Representations](https://arxiv.org/abs/2103.00020)
- [Few-shot Learning for Semantic Segmentation](https://arxiv.org/abs/1709.03410)

## Contributing

Please read our contributing guidelines and code of conduct before submitting pull requests.

## License

MIT License - see LICENSE file for details. 