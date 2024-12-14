# Weather Prediction Project

This project utilizes machine learning and deep learning to predict weather conditions in Nagaland based on historical data. It includes data preprocessing, feature engineering, GRU model training, and evaluation.

## Directory Structure
- **`data/`**  
  Contains the datasets used for training, testing, and evaluation:
  - `train.csv`: Training dataset
  - `test.csv`: Testing dataset
  - `evaluate.csv`: Dataset for final evaluation

- **`src/`**  
  Contains Python scripts for various stages of the project pipeline:
  - `data_preprocessing.py`: Handles data cleaning, encoding, and scaling
  - `feature_engineering.py`: Adds cyclic features and creates sequences for time-series modeling
  - `model_building.py`: Contains the model-building logic (GRU model)
  - `model_training.py`: Includes training and hyperparameter tuning logic
  - `evaluate_model.py`: Evaluates the model on test data
  - `utils.py`: Helper functions for logging and miscellaneous tasks

- **`model/`**  
  Directory to save the trained model:
  - `gru_model_weather_forecasting.h5`: Trained GRU model

- **`requirements.txt`**  
  Lists the dependencies required for the project.

- **`main.py`**  
  Main script to run the entire pipeline.

Kaggle Link for Notebook: https://www.kaggle.com/code/chetan0361/weather-predictions
