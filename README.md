# Weather Prediction Project 🌦️

This project predicts weather conditions in Nagaland using machine learning and deep learning techniques. It involves data preprocessing, feature engineering, and training a GRU-based model for time-series forecasting.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Directory Structure](#directory-structure)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Pipeline Details](#pipeline-details)
7. [Results](#results)
8. [Contributing](#contributing)
9. [License](#license)

---

## Project Overview

This project leverages historical weather data to predict daily weather conditions. The workflow includes:
- Cleaning and preprocessing weather data.
- Adding meaningful features using feature engineering techniques.
- Training a Gated Recurrent Unit (GRU) model for sequence-based predictions.
- Evaluating the trained model on unseen test data.

The GRU model is fine-tuned using Keras Tuner for optimal performance, ensuring accurate predictions.

---

## Requirements

Ensure you have the following installed:
- Python >= 3.7
- Required Python libraries listed in `requirements.txt`

---

## Installation

1. Clone the repository:
  git clone https://github.com/Chetan3110/weather-forecast.git
  cd weather-forecast

2. Create a virtual environment and activate it:
  python -m venv venv
  source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install the dependencies:
  pip install -r requirements.txt

---

## Usage
  
  Prepare the data: Ensure the train.csv and test.csv datasets are in the data/ directory.
  
  Run the pipeline: Execute the following command:
    python main.py
  
  View Results: The script will display the evaluation metrics (e.g., MAE) for the model.

---

## Pipeline Details

1. Data Preprocessing (data_preprocessing.py)
Cleans data by removing missing values.
Encodes categorical variables using LabelEncoder.
Scales features using MinMaxScaler.

2. Feature Engineering (feature_engineering.py)
Adds cyclic features (month_sin, month_cos) to capture seasonality.
Creates sequences for time-series modeling.

3. Model Building (model_building.py)
Defines a GRU-based deep learning model.
Utilizes Keras Tuner for hyperparameter optimization.

4. Model Training (model_training.py)
Trains the GRU model using the best hyperparameters from Keras Tuner.
Implements early stopping to prevent overfitting.

5. Evaluation (evaluate_model.py)
Evaluates the trained model on the test dataset.
Reports metrics like Mean Absolute Error (MAE).

---

## References

This project is inspired by the Kaggle notebook: [Weather Predictions Notebook](https://www.kaggle.com/code/chetan0361/weather-predictions/notebook).

The dataset used is available at: [Weather Predictions Dataset](https://www.kaggle.com/datasets/chetan0361/weather-predictions-dataset).
