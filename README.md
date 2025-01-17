# Flower Emotion Classification Project

## Overview

This project is designed to classify human-drawn flowers into one of three categories: **happy**, **sad**, or **angry**. The goal is to provide an intuitive user experience for drawing and labeling flowers, training a machine learning model, and predicting flower emotions.

The project includes the following components:
1. A **Flask web application** (`app.py`) where users can draw flowers on a blank canvas and get predictions.
2. A **labeling application** that allows users to draw flowers and label them as happy, sad, or angry. The labeled images are saved in the `my_flowers` directory.
3. A fully trained Convolutional Neural Network (CNN) model to classify the flowers.
4. Tools for **data augmentation** and **fine-tuning** the model on new data.

---

## Directory Structure

```
.
├── app.py                   # Flask application for prediction
├── augmented.py             # Data augmentation script
├── data_generation.py       # Script for generating dataset
├── flowers/                 # Original dataset folder
├── flowers_augmented/       # Augmented dataset folder
├── my_flowers/              # User-labeled flowers folder (created by the labeling app)
├── flower_model.pth         # Pre-trained model file
├── main_training.py         # Main training script
├── testing.py               # Model testing script
├── templates/               # HTML templates for the Flask app
└── README.md                # Documentation
```

---

## Features

### Flask Application (`app.py`)
- Allows users to **draw a flower** on a canvas and get its emotion predicted as **happy**, **sad**, or **angry**.
- Simple and interactive interface.

### Labeling Application
- Provides a **drawing canvas** for users to create flowers.
- Users can label their flowers and save them to the appropriate category folder in the `my_flowers` directory.
- Images are automatically resized to **256x256 pixels** and converted to grayscale.

### Data Augmentation
- Augments the dataset with techniques such as rotation, resizing, random erasing, and zooming.
- Ensures better generalization of the model by simulating diverse drawing styles.

### Training and Fine-tuning
- The CNN model (`main_training.py`) is pre-trained on a dataset of 100 flowers per category.
- The model can be fine-tuned with additional user-labeled data from the `my_flowers` directory.
- Training and validation accuracy/loss metrics are visualized using plots (see **Performance Metrics**).

---

## Performance Metrics

Below is the training and validation performance of the CNN model:

![Performance Metrics](performance.png)

**Observations:**
- The training loss consistently decreases, indicating proper learning.
- Validation accuracy stabilizes at a high percentage (~86%), showing good generalization.

---

## How to Use

### Prerequisites
1. Python 3.9 or above.
2. Install the required libraries using:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure you have the following folder structure:
   ```
   flowers/
     ├── train/
     │   ├── happy/
     │   ├── sad/
     │   └── angry/
     └── valid/
         ├── happy/
         ├── sad/
         └── angry/
   ```

### Running the Flask Application
1. Start the application:
   ```bash
   python app.py
   ```
2. Open your browser and navigate to `http://127.0.0.1:5000`.
3. Draw a flower on the canvas and click "Predict" to see the classification.

### Using the Labeling Application
1. Run the labeling script:
   ```bash
   python labeling_app.py
   ```
2. Draw flowers on the canvas and label them as **happy**, **sad**, or **angry**.
3. Click "Finish" to save the labeled flowers in the `my_flowers` directory.

### Fine-tuning the Model
1. Add new labeled data to the `my_flowers` directory.
2. Run the training script with the additional data:
   ```bash
   python main_training.py
   ```
3. Save the updated model for future use.

---






