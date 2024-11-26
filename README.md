# CXLNET: X-ray and CT Image Segmentation with Novel Neural Network Model

This project demonstrates the application of a novel neural network model for image segmentation tasks on X-ray and CT scan images. The goal is to improve the segmentation accuracy of medical images, which can aid in diagnosing conditions such as pneumonia, lung diseases, and other abnormalities visible in X-ray or CT images.

The repository includes all necessary code to load, preprocess, train, and evaluate a model for semantic segmentation. The core model is implemented using state-of-the-art deep learning techniques, and the dataset is carefully split and managed for training and testing purposes.

## Project Overview

Medical image segmentation is a critical task in the field of healthcare, as it helps in identifying and isolating specific areas of interest from medical images. This project leverages deep learning to perform efficient segmentation of X-ray and CT images using a custom neural network architecture.

Key components of the repository include:
- **Data preprocessing**: Code for loading and preprocessing X-ray and CT images.
- **Neural network architecture**: A custom model designed to handle image segmentation tasks.
- **Training and evaluation**: Jupyter notebooks to train the model and evaluate its performance.
- **Dataset split**: The dataset is divided into training and testing sets using a split configuration file (`split.pk` and `splits_CT.pk`).

## Key Files

- **`data.py`**: Contains functions for loading and preprocessing the dataset. It handles tasks like resizing images, normalizing pixel values, and augmenting the data for training.
  
- **`OurModel.py`**: Defines the architecture of the deep learning model used for image segmentation. It includes convolutional layers, pooling, and upsampling operations to produce accurate segmentations of X-ray and CT images.
  
- **`training.ipynb`**: Jupyter notebook for training the neural network model on the dataset. It contains all the necessary code to load the data, initialize the model, and start the training process, along with visualizations of training progress.
  
- **`evaluate.ipynb`**: Jupyter notebook for evaluating the trained model on the test set. It provides performance metrics, such as accuracy, intersection over union (IoU), and visual outputs of segmented images compared to the ground truth.
  
- **`split.pk and splits_CT.pk`**: A pickle file used to manage the dataset split. It ensures that the data is divided correctly for training and testing, allowing reproducibility of the results.

## Installation

To set up the project locally, clone the repository and install the required dependencies:

```bash
git clone https://github.com/your-username/CXLNET.git
cd xray-ct-image-segmentation
pip install -r requirements.txt
```

## Requirements

The following libraries are required to run the project:

- `tensorflow` (for deep learning model implementation)
- `keras` (for high-level neural network API)
- `numpy` (for numerical computations)
- `scikit-learn` (for performance evaluation)
- `matplotlib` (for visualizations)

Install all dependencies using:

```bash
pip install -r requirements.txt
```

## Usage

### Training the Model
To train the model, open the `training.ipynb` file in Jupyter Notebook or Google Colab. Follow the instructions in the notebook to preprocess the data, configure the model, and start the training process. You can monitor the progress through loss and accuracy plots during training.

### Evaluating the Model
After training, open the `evaluate.ipynb` file to assess the performance of the trained model on unseen data. This notebook will display metrics like accuracy and IoU (Intersection over Union), and it will visualize the segmented images compared to the ground truth.

## Model Details

The model architecture used in this project is designed to work effectively with medical images. It employs convolutional neural networks (CNNs) and uses an encoder-decoder structure with skip connections to ensure that fine-grained details are preserved in the output segmentation maps.

