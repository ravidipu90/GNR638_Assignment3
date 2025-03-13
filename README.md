# UC Merced Land Use Classification using CNN & Grad-CAM Visualization

## Overview
This project trains a small Convolutional Neural Network (CNN) from scratch to classify images from the **UC Merced Land Use Dataset**. After training, we implement **Class Activation Mapping (CAM)** using **Grad-CAM** to visualize which parts of an image contribute the most to the model's decision.

## Dataset
- **UC Merced Land Use Dataset** consists of **21 land-use categories**, with 100 images per class.
- Images are **256x256 pixels**.
- The dataset is expected to be stored under `data/<class_name>/*.tif or *.jpg`.

## Model Architecture
The CNN model consists of:
- **Three convolutional blocks**, each with:
  - **Conv2D layers** with ReLU activation
  - **Batch Normalization** for stable training
  - **Max Pooling** for downsampling
  - **Dropout** for regularization
- **Global Average Pooling (GAP)** to reduce spatial dimensions
- **Fully connected dense layer** with dropout
- **Softmax activation for multi-class classification**

## Data Augmentation
To improve generalization, the following augmentations are applied:
- **Horizontal Flip**
- **Random Rotation (15 degrees)**
- **Zoom Range (20%)**
- **Width & Height Shift (10%)**
- **Rescaling (1./255 normalization)**

## Training Details
- Optimizer: **Adam (learning rate = 1e-3)**
- Loss Function: **Categorical Crossentropy**
- Metrics: **Accuracy**
- Callbacks:
  - **EarlyStopping** (patience=10, restores best weights)
  - **ModelCheckpoint** (saves best model based on validation accuracy)
  - **ReduceLROnPlateau** (reduces LR if validation loss plateaus)
- Training for **100 epochs** with a **batch size of 32**

## Grad-CAM Visualization
- Grad-CAM helps highlight **important regions** in an image contributing to a classification decision.
- We extract gradients of the **last convolutional layer** with respect to the output class.
- Heatmaps are overlayed on the original image to visualize important areas.

## Installation & Usage
### Prerequisites
Ensure you have the following installed:
- Python 3.7+
- TensorFlow 2.x
- OpenCV
- Matplotlib
- NumPy

### Steps to Run the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/ucmerced-cnn-gradcam.git
   cd ucmerced-cnn-gradcam
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place the **UC Merced dataset** inside `data/`.
4. Run the training script:
   ```bash
   python train.py
   ```
5. View **Grad-CAM visualizations** by running:
   ```bash
   python visualize_gradcam.py
   ```

## Example Results
Below are Grad-CAM visualizations for sample images:

| Original Image | Grad-CAM Heatmap | Overlayed Image |
|---------------|-----------------|-----------------|
| ![Original](sample_original.jpg) | ![Heatmap](sample_heatmap.jpg) | ![Overlayed](sample_overlay.jpg) |

## Acknowledgments
- UC Merced Land Use Dataset
- Grad-CAM: **Selvaraju et al. (2017)**

---
**Author:** Your Name  
**License:** MIT  

