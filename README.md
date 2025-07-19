# 🧬 Rabies Cell Image Classification using Deep Learning

This project aims to classify microscopy images of rabies-infected cells using state-of-the-art deep learning techniques. The objective is to support biological research by automating the identification of infection signs within cell samples.

## 🔍 Problem Statement

Detect and classify wells containing rabies-infected cells in microscopy images. Given the small size of the dataset, classical machine learning models suffered from overfitting. To overcome this, a robust deep learning pipeline using **transfer learning** and **object detection** was developed.

---

## 🧪 Methodology

### 1. 🔧 Image Annotation and Well Detection (YOLO + Roboflow)
- Raw microscopy images often contain multiple wells.
- Annotated manually via **Roboflow** to train a **YOLOv5** model to detect and extract well regions from the full images.
- This enabled us to isolate the meaningful regions for classification.

### 2. 🧠 Deep Learning Pipeline with Transfer Learning
- Multiple pretrained CNN models were evaluated:
  - **EfficientNet**
  - **ResNet**
  - **MobileNet**
- Applied **data augmentation** to improve generalization (flip, brightness, rotation, contrast, etc.).
- Used **transfer learning** to fine-tune the models on our domain-specific data.
- Experiments were tracked using **TensorBoard** for:
  - Training/validation loss
  - Accuracy
  - Overfitting diagnosis
  - Confusion matrices

### 3. 🚀 Deployment
- The best model (EfficientNet-based) was selected based on validation accuracy and generalization.
- Deployed using **Gradio** and shared publicly on **Hugging Face Spaces**:

👉 [🔗 Live Demo on Hugging Face](https://huggingface.co/spaces/huggingkhalil/efficientnet-classifier)

---

## 📊 Why Deep Learning was Better than Classical ML

Classical machine learning models like Random Forest, SVM, and Logistic Regression showed signs of **overfitting** due to the small dataset:

| Model              | Training Score | Validation Score | Conclusion              |
|-------------------|----------------|------------------|--------------------------|
| Random Forest      | 1.0            | Low, fluctuating | Strong overfitting       |
| SVM                | Near 1.0       | Improving        | Moderate overfitting     |
| Logistic Regression| 1.0            | Stable           | Slight overfitting       |
| Gradient Boosting  | 1.0            | Unstable         | High variance / overfit  |

By contrast, **transfer learning** with CNNs (especially EfficientNet) yielded **much better generalization**, and **TensorBoard** helped to monitor and choose the optimal model configuration.

---

## 🤝 Acknowledgments

Special thanks to:

- **Mariem Handous** and **Ines Abdeljaoued-Tej** for their supervision and guidance.
- **Le laboratoire BIMS (LR16IPT09)** and **le laboratoire de Rage de l’Institut Pasteur de Tunis** for providing the dataset and hosting the project.


