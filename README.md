# Humidity Prediction using Machine Learning and Deep Learning

This project focuses on predicting humidity levels using both machine learning (ML) and deep learning (DL) techniques. Accurate humidity prediction is essential for various applications, including weather forecasting, agriculture, and climate studies.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [License](#license)

## Overview

The project employs the `AirQuality.csv` dataset, which contains hourly measurements of various weather attributes, including temperature, humidity, air pressure, and other meteorological factors. The goal is to develop predictive models that can forecast humidity levels based on these features.

**Key Steps in the Analysis:**

1. **Data Preprocessing:**
   - Load and clean the dataset.
   - Handle missing values and outliers.
   - Split the data into training and testing sets.

2. **Feature Engineering:**
   - Select relevant features that influence humidity.
   - Engineer new features if necessary to improve model performance.

3. **Model Building:**
   - **Machine Learning Models:**
     - Implement algorithms such as Linear Regression, Decision Tree Regression, and Random Forest Regression.
   - **Deep Learning Models:**
     - Construct neural networks using frameworks like TensorFlow or Keras.
     - Define architectures suitable for time-series forecasting if temporal patterns are present.

4. **Model Training:**
   - Train both ML and DL models on the training dataset.
   - Tune hyperparameters to optimize performance.

5. **Model Evaluation:**
   - Evaluate models using metrics such as Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R-squared.
   - Compare the performance of ML and DL models to determine the most effective approach.

6. **Prediction:**
   - Use the best-performing model to predict humidity levels on new or unseen data.

## Project Structure

The repository contains the following files:

- `AirQuality.csv`: The dataset used for training and testing the models.
- `humidity-prediction.ipynb`: A Jupyter Notebook that includes code for data preprocessing, feature engineering, model building, training, and evaluation.

## Setup Instructions

To set up and run the project locally, follow these steps:

1. **Clone the Repository:**
   Use the following command to clone the repository to your local machine:

   ```bash
   git clone https://github.com/dattatejaofficial/Humidity-Prediction-using-ML-DL.git
   ```

2. **Navigate to the Project Directory:**
   Move into the project directory:

   ```bash
   cd Humidity-Prediction-using-ML-DL
   ```

3. **Create a Virtual Environment (optional but recommended):**
   Set up a virtual environment to manage project dependencies:

   ```bash
   python3 -m venv env
   ```

   Activate the virtual environment:

   - On Windows:
     ```bash
     .\env\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source env/bin/activate
     ```

4. **Install Dependencies:**
   Install the required Python packages using pip:

   ```bash
   pip install pandas scikit-learn tensorflow matplotlib seaborn
   ```

## Usage

To run the analysis:

1. **Ensure the Virtual Environment is Activated:**
   Make sure your virtual environment is active (refer to the setup instructions above).

2. **Open the Jupyter Notebook:**
   Launch Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

   Open `humidity-prediction.ipynb` in the Jupyter interface and execute the cells sequentially to perform the analysis.

## Dependencies

The project requires the following Python packages:

- `pandas`
- `scikit-learn`
- `tensorflow`
- `matplotlib`
- `seaborn`

These dependencies are essential for data manipulation, model building, and visualization.
