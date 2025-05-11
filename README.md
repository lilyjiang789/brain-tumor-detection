# Brain Tumor Classification using CNN

This project is a deep learning-based multiclass classification system that uses **Convolutional Neural Networks (CNNs)** to classify brain MRI images into one of four tumor types. It leverages the [Kaggle Brain Tumor Classification (MRI)](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri) dataset and achieves a test accuracy of **74%**.

---

## Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 1.00      | 0.19   | 0.32     | 100     |
| 1     | 0.77      | 1.00   | 0.87     | 105     |
| 2     | 0.61      | 1.00   | 0.76     | 115     |
| 3     | 1.00      | 0.70   | 0.83     | 74      |

- **Overall Accuracy**: 74%  
- **Macro Avg F1-Score**: 0.69  
- **Weighted Avg F1-Score**: 0.69  

While classes 1 and 2 are predicted effectively, **Class 0** has high precision but low recall, indicating further tuning or data balancing is needed for improved performance.

---

## Project Overview
* **Goal**: Classify input data into one of four medical categories using supervised learning.

* **Approach**: Preprocessing, model training, evaluation, and report generation.

* **Tech Stack**: Python, TensorFlow, NumPy, pandas, matplotlib


## Dataset

- **Source**: [Kaggle - Brain Tumor Classification (MRI)](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)
- **Classes**:
  - **0**: Glioma Tumor  
  - **1**: Meningioma Tumor  
  - **2**: No Tumor  
  - **3**: Pituitary Tumor

---

## Model Architecture

- **Model**: Convolutional Neural Network (CNN)
- **Framework**: TensorFlow / Keras
- **Input Size**: Resized MRI images (e.g., 150x150 or 224x224)
- **Layers**:
  - Convolutional Layers with ReLU
  - MaxPooling
  - Fully Connected Dense Layers
  - Dropout for regularization
  - Softmax output for multiclass classification

---

## Getting Started

### Requirements

- Python 3.8+
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

### Installation

```bash
git clone https://github.com/yourusername/brain-tumor-cnn.git
cd brain-tumor-cnn
pip install -r requirements.txt
```


---

## Results & Improvements

- Best results achieved with **data augmentation** and **dropout layers**.
- Future improvements:
  - Address class imbalance (especially for Class 0)
  - Try transfer learning (e.g., ResNet, VGG16)
  - Incorporate explainability (Grad-CAM, SHAP)



