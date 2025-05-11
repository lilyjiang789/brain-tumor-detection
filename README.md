# Brain Tumor Classification using EfficientNetB0 (Transfer Learning)

This project is a deep learning-based multiclass classification system that uses **EfficientNetB0** via transfer learning to classify brain MRI images into one of four tumor types. It uses the [Kaggle Brain Tumor Classification (MRI)](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri) dataset and achieves a test accuracy of **74%** with class weighting, callbacks, and augmentation.

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

---

## Dataset

- **Source**: [Kaggle - Brain Tumor Classification (MRI)](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)
- **Classes**:
  - **0**: Glioma Tumor  
  - **1**: Meningioma Tumor  
  - **2**: No Tumor  
  - **3**: Pituitary Tumor

---

## Model Architecture

- **Base Model**: `EfficientNetB0` (pretrained on ImageNet, `include_top=False`)
- **Architecture**:
  - Input shape: `(image_size, image_size, 3)`
  - GlobalAveragePooling2D
  - Dropout (rate = 0.5)
  - Dense output layer with 4 units (softmax activation)
- **Loss Function**: `categorical_crossentropy`
- **Optimizer**: `Adam`
- **Metrics**: `accuracy`

---

## Training Strategy

- **Callbacks Used**:
  - `EarlyStopping` (optional, recommend adding)
  - `ReduceLROnPlateau` (monitoring `val_accuracy`)
  - `ModelCheckpoint` (save best model)
  - `TensorBoard` logging

- **Class Imbalance Handling**:
  - Used `sklearn.utils.class_weight.compute_class_weight` to compute and apply `class_weight` during training.

- **Validation**:
  - `validation_split=0.1` used from training set

- **Epochs**: 12  
- **Batch Size**: 32  

---

## Getting Started

### Requirements

- Python 3.8+
- TensorFlow / Keras
- NumPy
- Scikit-learn
- Matplotlib (optional)

### Installation

```bash
git clone https://github.com/yourusername/brain-tumor-efficientnet.git
cd brain-tumor-efficientnet
pip install -r requirements.txt
```

### Run the Model

```python
python train_effnet.py
```

---

## Potential Improvements

- Use image augmentation (e.g., `ImageDataGenerator` or `tf.keras.preprocessing`)
- Further fine-tune EfficientNet layers (currently frozen by default)
- Apply Grad-CAM to visualize decision regions
- Try other EfficientNet variants (e.g., B1–B3) or pretrained models

---

## Contributing

Feel free to fork the repo, open issues, or suggest improvements!

---

## License

This project is licensed under the MIT License.
