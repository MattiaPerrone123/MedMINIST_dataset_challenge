# MedMNIST_dataset_challenge

This repository contains an end-to-end medical imaging pipeline for classifying images from the [MedMNIST dataset](https://github.com/MedMNIST/MedMNIST). The PathMNIST dataset is used for the current analysis

## Project Structure
```
MedMNIST_dataset_challenge/
│
│
├── output/                                         # Folder with generated outputs 
│   ├── pathmnist/                                   # Folder with the name of the dataset used
│     ├── loss_plot.png                              # Plot of training and validation loss
│     ├── pathmnist_log.txt                          # Log of train, val and test auc, accuracy and f1-score
│     ├── Tensorboard_Results/      
│         ├── events.out.tfevents.1730321268         # Record of training related information
│     ├── confusion_matrixes/      
│         ├── train_confusion_matrix.png             # Confusion_matrix on training set
│         ├── val_confusion_matrix.png               # Confusion_matrix on validation set
│         ├── test_confusion_matrix.png              # Confusion_matrix on test set
│
├── models/             
│   ├── __init__.py
│   ├── initializer.py                               # Functions for initializing and configuring models
│   ├── models.py                                    # Defines model architectures (custom ResNet)
│
├── utils/              
│   ├── __init__.py
│   ├── device_setup.py                              # Functions for setting up the device
│   ├── data_loader.py                               # Functions to load and transform datasets
│   ├── training_testing.py                          # Functions for training, evaluating, and logging metrics
│   ├── model_io.py                                  # Functions for model management
│   ├── visualization.py                             # Functions for plotting and saving visual outputs
│
├── main.py                                          # Main script for model training and evaluation
│
├── medmnist_analysis.py                             # Example of model training and evaluation
│
├── requirements.txt                                 # List of package dependencies
│
├── .gitignore                                       # Files to exclude from version control
│
└── README.md                                        # Project overview and instructions
```
## Setup Instructions

### Prerequisites

- **Python 3.11 or higher**
- Recommended: **GPU support** for faster training with CUDA 

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/MattiaPerrone123/MedMNIST_dataset_challenge.git
   cd MedMNIST_dataset_challenge
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   

## Usage Instructions

### Model Training and Evaluation

To see an example of model training and evaluation, refer to the `medmnist_analysis.ipynb` notebook. 
The model achieving the highest validation AUC will have its weights saved in the output/pathmnist directory. For reference, refer to this [link](https://drive.google.com/file/d/1ahbDq5CyOlMmOJCPSr0xkpd3VFSZewzT/view?usp=sharing).




## Parameters

- **data_flag**: Specifies the MedMNIST dataset to use (e.g., `'pathmnist'`)
- **output_root**: Directory to store output files
- **num_epochs**: Number of training epochs
- **gpu_ids**: GPU identifier for CUDA support (e.g., `'0'`)
- **batch_size**: Number of samples per batch during training
- **lr**: Learning rate for the optimizer
- **gamma**: Decay factor for learning rate scheduling
- **size**: Size of images in the dataset
- **download**: Set to `True` to download the dataset if not available
- **model_flag**: Model type (e.g., `'resnet18'`)
- **resize**: Set to `True` to resize the images to 224x224 for torchvision models
- **as_rgb**: Set to `True` to use images in RGB format
- **run**: Name of the experiment run
- **confusion_matrix**: Set to `True` to generate confusion matrix plots
- **plot_epochs**: Set to `True` to generate loss plots over epochs
- **optimizer_type**: Optimizer selection (`'adam'` or `'sgd'`)


## Additional Notes

### Starting Code
This project builds upon the [MedMNIST project](https://github.com/MedMNIST/MedMNIST).

### Project Constraints

- **`run` parameter**: Modifying or removing this parameter requires analysis and adaptation of the `evaluate` function from the MedMNIST library, which was skipped because of time constraints
- **Pre-defined Models**: Pre-configured models (e.g., ResNet) cannot be embedded directly in the `main` function without additional setup. However, this can be easily adapted for a more streamlined workflow.
- **Resize to 224x224**: The resizing option to 224x224 was chosen for compatibility with torchvision models like ResNet and since already implemented in the MedMNIST repo. For custom sizes, consider verifying the model’s architecture compatibility, as some dimensions may cause issues.
- **Image Size**: The `size` parameter is dependent on the MedMNIST library, which determines the resolution of images used in the dataset.

### Suggested Next Steps

- **Set Random Seed**: For reproducibility, adding a random seed to the pipeline would be beneficial
- **Hyperparameter Tuning and Data Augmentation**: These optimizations were deprioritized in this version due to time constraints and current focus on functionality over model performance. However, they would likely enhance the model's accuracy and robustness
- **Image Visualizations**: Implementing a feature to visualize misclassified samples can provide valuable insights for error analysis




