# Trapped Ion State Classification with Semi-Supervised Neural Networks

## Overview

This project implements a novel neural network architecture for classifying quantum states in trapped ions using fluorescence images. The model employs both supervised and semi-supervised learning approaches to achieve high classification accuracy while reducing dependence on manually labeled data.

![Neural Network Architecture](model_architecture.png)

## Key Features

- **Custom Neural Network Architecture**: An encoder-decoder structure with index-dependent dense layers tailored for ion state processing
- **Semi-Supervised Learning**: Reduces labeled data requirements by 80% through innovative training techniques
- **Correlation Loss Function**: Custom implementation to ensure uniform distribution of predicted states
- **Quantum State Classification**: Achieves high accuracy in classifying fluorescence ion states across 16 distinct classes
- **FPGA Deployment Ready**: Model quantization enables real-time deployment on specialized hardware

## Technical Background

Trapped ions are quantum systems that can be manipulated to exist in discrete states. The 4-ion system in this project can exist in 16 (2⁴) possible states. While detecting these states directly is challenging, we can capture fluorescence images of the ions and use machine learning to classify them.

The challenge addressed in this project is how to effectively train models when labeled data is limited and expensive to obtain. Our approach:

1. First train a base model on a small labeled dataset
2. Enhance the model with unlabeled data using a novel correlation loss function
3. Drive predictions toward a uniform distribution over all possible states

## Architecture

The neural network consists of:

- **Encoder**: Three stacked IndexDependentDense layers that extract features from fluorescence images
- **Classifier**: Single IndexDependentDense layer with log_softmax activation for state classification
- **Coupler**: Custom module that enhances the model to better handle unlabeled data
- **Gumbel-Softmax**: Enables differentiable sampling for discrete state prediction

## Installation

```bash
# Clone the repository
git clone https://github.com/SeivenBell/Trapped_Ion_FFNN_Classifier.git
cd Trapped_Ion_FFNN_Classifier

# Install dependencies
bash install_dependencies.sh
```

## Dataset Structure

The project requires two types of datasets:
- **Labeled Data**: Ion fluorescence images with corresponding quantum state labels
- **Unlabeled Data**: Ion fluorescence images without labels (halfpi dataset)

Both datasets should be in HDF5 format with the following structure:
- Labeled data: 'measurements' and 'labels' datasets
- Unlabeled data: 'measurements' dataset

## Usage

### Training the Base Model

```python
from classifier.main import main

# Run the main training process
main()
```

### Configuration

Adjust training parameters in `classifier/config.py`:

```python
config = {
    "full_dataset_path": "path/to/labeled_data.h5",
    "halfpi_dataset_path": "path/to/unlabeled_data.h5",
    "batch_size": 250,
    "val_ratio": 0.2,
    "train_params": {"N_epochs": 25, "lr": 0.00035},
    "enhanced_train_params": {
        "N_epochs": 15,
        "lr": 0.00016,
        "weight_decay": 5.6e-5,
    },
    "device": "cpu",  # Or "cuda" for GPU training
    "log_every": 1,
}
```

## Results

The semi-supervised approach demonstrates significant improvements:

- **Base Model**: Achieves ~91-92% accuracy on labeled data
- **Enhanced Model**: Produces more uniform state distributions across unlabeled data
- **Data Efficiency**: Reduces labeled data requirements by up to 80%
- **Performance Metric**: Sum of distances to uniform distribution reduced from ~0.52 to ~0.36

The model effectively learns to identify all 16 possible quantum states in the 4-ion system, even with limited labeled examples.

## How It Works

1. **Base Training Phase**: Train the initial model with supervised learning on labeled data
2. **Enhancement Phase**: Freeze the base model and train a coupler component using correlation loss on unlabeled data
3. **Correlation Loss**: Forces the model to produce decorrelated outputs across ions, encouraging uniform state distribution
4. **Model Composition**: The final model combines the pre-trained encoder-decoder with the optimized coupler

## License

This project is available under the MIT License.

## Acknowledgments

This research builds on work from the Institute for Quantum Computing and the University of Waterloo Quantum Information Group.

## Contact

For questions or collaboration opportunities, contact: sbalaniu@uwaterloo.ca
