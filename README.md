# NYCU Selected Topics in Visual Recognition using Deep Learning 2025 Spring HW1

StudentID: 110550129

Name: 姚咨妙

## Introduction
This project focuses on classifying images of 100 different plant and animal species using deep learning. Due to the imbalanced nature of the dataset, special loss functions such as **class weighting** and **focal loss** are explored to improve model performance. The model used is **ResNeXt50-32x4d**, which offers better performance under parameter constraints. Various hyperparameter tuning strategies, data augmentation techniques, and loss functions are tested to optimize accuracy.

## Installation
To set up and run this project, follow these steps:

### Prerequisites
Ensure you have Python installed along with PyTorch and required dependencies.

### Steps
1. Download or git clone this project

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the training script:
   ```bash
   python HW1.py
   ```

5. To test the best model and generate predictions:
   ```bash
   python predict.py
   ```

## Performance Snapshot

<img src="https://github.com/yuiolyzm/Selected-Topics-in-Visual-Recognition-using-Deep-Learning/blob/main/img/accuracy_curve.png" alt="drawing" width="200"/>
![image](https://github.com/yuiolyzm/Selected-Topics-in-Visual-Recognition-using-Deep-Learning/blob/main/img/accuracy_curve.png)

