# WiDS-Traffic-Sign-Classifier
Building a CNN model to classify traffic signs for autonomous vehicle perception as part of WiDS 5.0, IIT Bombay.

# Project Progress — Week-by-Week

This document summarizes the week-by-week progress of the project, including data preparation, model design, training, and evaluation.

## Week 1 — Python Fundamentals & Environment Setup
- **Environment:** Set up development environments using Google Colab and Kaggle.
- **Learning:** Mastered basic Python syntax (variables, lists, functions).
- **Practice:** Solved algorithmic problems (e.g., Two Sum) to improve logical problem solving.

## Week 2 — Machine Learning Foundations & Pandas
- **Theory:** Studied Linear Regression, cost functions, and gradient descent.
- **Data Handling:** Used Pandas for data cleaning and manipulation (useful for datasets such as GTSRB).
- **Neural Networks:** Explored TensorFlow Playground to visualize weight and bias adjustments.

## Week 3 — From Theory to Code (Data Preprocessing)
- **Image Processing:** Used Pillow (PIL) and NumPy to load ~50,000+ images and resize them to a uniform $30 \times 30$ resolution.
- **Normalization:** Scaled pixel values by dividing by `255.0`.
- **Data Splitting:** Created a validation split (20%) using `train_test_split` to monitor performance on unseen data.

## Week 4 — Convolutional Neural Networks (CNN)
- **Architecture:** Built a CNN using the Keras `Sequential` API.
- **Layers:** Added `Conv2D` layers for feature extraction and `MaxPooling` to reduce spatial dimensions.
- **Regularization:** Applied `Dropout` (0.25, 0.5) to reduce overfitting.
- **Classification:** Final `Dense` layer with `softmax` activation for 43-class classification.
- **Training:** Trained for 15 epochs with the Adam optimizer and categorical crossentropy loss.

## Week 5 — Final Evaluation
- **Performance:** Achieved a test accuracy of **97.17%**.
- **Visualization:** Plotted accuracy and loss over epochs to check for stability and overfitting.
- **Model Export:** Saved final weights in `.h5` format for deployment.

## 🛠️ Tech Stack
- **Language:** Python
- **Libraries:** TensorFlow, Keras, Scikit-learn, NumPy, Pandas, Matplotlib, PIL
- **Hardware:** Lenovo Legion 5 (RTX 4070 GPU) for local experimentation

## 📝 Reflection & Acknowledgments

Due to an unexpected family medical emergency, the project timeline was accelerated toward the end. Despite this, the pipeline was implemented and delivered with strong results.

- **Tools & Mentorship:** Utilized AI tools for debugging and consulted seniors for guidance on CNN architecture.
- **Current Status:** Pipeline implemented and evaluated; continuing to deepen understanding of TensorFlow and Scikit-learn internals.
- **Goal:** Treat this project as a foundation for further study in Computer Vision beyond WiDS 5.0.


## 🔗 Project Links

- [Week 3 Google Colab Notebook (Linear Regression & Neural Network)](https://colab.research.google.com/drive/1vyptkj6B5_vC1bczxqLLQLkIGon1M_Td?usp=sharing)
- [Week 4 Colab Notebook (The "AI Eye" for Handwritten Digits)](https://colab.research.google.com/drive/1QNIu577lX3q1tuZhzKZasrXS_iKH4I2l?usp=sharing)
- [Final Project Link ](https://www.kaggle.com/code/codeacc/final-assignment)



