# ✍️ Handwritten Character Recognition
### CodeAlpha Machine Learning Internship — Task 3

![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?style=flat-square&logo=streamlit)
![MNIST](https://img.shields.io/badge/MNIST-99.51%25-brightgreen?style=flat-square)
![EMNIST](https://img.shields.io/badge/EMNIST-95.30%25-brightgreen?style=flat-square)

---

## 📌 Overview

A deep learning project that recognizes handwritten digits (0–9) and letters (A–Z) using Convolutional Neural Networks (CNN) trained on the **MNIST** and **EMNIST Letters** datasets. Includes an interactive **Streamlit dashboard** for real-time predictions.

---

## 🎯 Results

| Dataset | Test Accuracy | Loss | ROC-AUC |
|---------|:-------------:|:----:|:-------:|
| MNIST (Digits 0–9) | **99.51%** | 0.0141 | **1.0000** |
| EMNIST (Letters A–Z) | **95.30%** | 0.1358 | **0.9989** |

---

## 🏗️ Model Architecture

### MNIST — Digits (0–9)
```
Input (28×28×1)
  → Conv2D(32, 3×3) + BatchNorm + MaxPool + Dropout(0.25)
  → Conv2D(64, 3×3) + BatchNorm + MaxPool + Dropout(0.25)
  → Flatten → Dense(256) + BatchNorm + Dropout(0.5)
  → Output: Dense(10, softmax)
```

### EMNIST — Letters (A–Z)
```
Input (28×28×1)
  → Conv2D(32, 3×3) + BatchNorm + MaxPool + Dropout(0.25)
  → Conv2D(64, 3×3) + BatchNorm + MaxPool + Dropout(0.25)
  → Conv2D(128, 3×3) + BatchNorm + Dropout(0.25)
  → Flatten → Dense(512) + BatchNorm + Dropout(0.5)
  →           Dense(256) + Dropout(0.3)
  → Output: Dense(26, softmax)
```

---

## 📁 Project Structure

```
CodeAlpha_HandwrittenCharacterRecognition/
│
├── data/                                           # Local datasets
│   ├── mnist.npz
│   ├── emnist-letters-train-images-idx3-ubyte.gz
│   ├── emnist-letters-train-labels-idx1-ubyte.gz
│   ├── emnist-letters-test-images-idx3-ubyte.gz
│   └── emnist-letters-test-labels-idx1-ubyte.gz
│
├── CodeAlpha_HandwrittenCharacterRecognition.ipynb # Training notebook
├── app.py                                          # Streamlit dashboard
├── requirements.txt                                # Dependencies
├── mnist_cnn.keras                                 # Trained MNIST model
├── emnist_cnn.keras                                # Trained EMNIST model
└── README.md
