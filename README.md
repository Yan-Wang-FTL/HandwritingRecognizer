# Advanced Machine Learning Algorithms on Motion-Based Handwriting Recognition

## 📖 Overview
This project explores **motion-based online handwriting recognition** using data from digital pens equipped with inertia sensors (accelerometers & gyroscopes).  
Unlike image-based OCR, this approach does not require cameras and is robust against lighting and occlusions.  

We compare **eight supervised machine learning algorithms** across two evaluation modes:
- **Random Split Mode** – data from all subjects shuffled into train/validation/test.  
- **Subject Split Mode** – data split within each subject individually.  

The study achieves **93% accuracy** in both modes using **k-Nearest Neighbors (k=1, Euclidean distance)**, surpassing previous benchmarks (86.67% with LSTM in Chen et al., 2019).

---

## 🧠 Algorithms Implemented
1. **k-Nearest Neighbors (k-NN)**  
2. **Softmax Regression Models**  
   - Linear Regression  
   - Artificial Neural Networks (ANN)  
   - Convolutional Neural Networks (CNN)  
   - Recurrent Neural Networks (LSTM, GRU)  
   - Custom Transformer-inspired architecture  
3. **Support Vector Machines (SVMs)** with Gaussian kernel and hyperparameter optimization  

---

## 📊 Dataset
- Based on [**Chen et al. (2019)**](https://arxiv.org/abs/2101.06022) handwriting dataset.  
- Data collected via **9-axis digital pen sensors**:  
  - Rotation signals: Yaw, Pitch, Roll  
  - Acceleration signals: ax, ay, az  
- **10,400 samples** from **20 subjects**, each writing 20 samples of all 26 English lowercase letters.  

---

## 🔧 Pre-processing Pipeline
1. **Calibration** – remove sensor offsets.  
2. **Interpolation** – normalize sequence lengths (100 samples per sequence).  
3. **Denoising** – moving average low-pass filter (n=8).  
4. **Normalization** – zero mean, unit variance.  
5. **Flattening** – 300-dimensional feature vector (3 channels × 100 timesteps).  

👉 **Ablation studies** confirm that **denoising** is the most critical step for accuracy.

---

## 🚀 Results
| Algorithm            | Random Split | Subject Split |
|----------------------|--------------|---------------|
| **k-NN (k=1)**       | **93%**      | **93%**       |
| ANN                  | 90%          | 93%           |
| CNN                  | 88%          | 89%           |
| LSTM                 | 81%          | 84%           |
| GRU                  | 87%          | 86%           |
| Custom Transformer   | 88%          | 91%           |
| SVM                  | 88%          | 94%           |

- **Best overall method**: k-NN (simple & efficient).  
- **SVMs** achieved strong subject-level performance (94%).  
- **Normalization** had limited effect, except in SVMs.  
- **Denoising** proved essential for all models.  

---

## 🏆 Key Contributions
- Improved recognition accuracy from **86.67% → 93%**.  
- Introduced a **Transformer-inspired classifier** for motion-based handwriting.  
- Performed **ablation studies** on preprocessing steps.  
- Demonstrated that **simple models (k-NN)** can outperform complex neural networks when combined with good preprocessing.  

---

## ▶️ Instructions to Run

1. **Create conda env with Python 3.13**  
    ```bash
    conda create -n new_env python=3.13
    conda activate new_env
2. **Install dependencies**
    ```bash
    pip install -r requirements.txt
3. **Run a model with python _model_name.py_**

    **Examples:**
    ```bash
    python ann.py
    python transformer.py
    python knn.py
    python svm.py
4. **Results will be saved to the *results/* directory**

## 📬 Contact
- **Yan Wang** – Department of Computer Science, University of Alberta  
- **Xueying Zhang** – Department of Mathematical and Statistical Science, University of Alberta  
