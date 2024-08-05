# Housing Prices Prediction

This repository contains a Jupyter Notebook for predicting housing prices using machine learning models. The notebook explores various stages of data processing, model building, and evaluation to predict housing prices accurately.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [Results](#results)

## Overview

The objective of this project is to predict housing prices based on various features of the houses. This is achieved by utilizing different machine learning techniques and models to identify patterns and correlations in the dataset.

## Dataset

The dataset used in this project is the [Housing Prices Dataset](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset/data) from Kaggle. This dataset includes various features such as the number of bedrooms, size in square feet, location, etc. Ensure that the dataset is available in your working directory or update the path accordingly in the notebook.

## Dependencies

To run the notebook, the following Python libraries are required:

- pandas
- numpy
- matplotlib
- scikit-learn
- tensorflow (for neural network models)

You can install these dependencies using pip:

```bash
pip install pandas numpy matplotlib scikit-learn tensorflow
```

## Usage

1. Clone this repository:

    ```bash
    git clone https://github.com/GaganBansal22/Housing-Prices-Prediction.git
    ```

2. Navigate to the project directory:

    ```bash
    cd housing-prices-prediction
    ```

3. Open the Jupyter Notebook:

    ```bash
    jupyter notebook Housing__Prices.ipynb
    ```

4. Execute the cells in the notebook sequentially to preprocess the data, build, train, and evaluate the models.

## Modeling

The notebook includes the following steps for modeling:

1. **Data Preprocessing**: Handling missing values, encoding categorical features, and normalizing the data.
2. **Exploratory Data Analysis (EDA)**: Visualizing data distributions and relationships between features.
3. **Model Building**:
    - Linear Regression
    - Support Vector
    - Decision Tree
    - Random Forest
    - CatBoost
    - Neural Network (using TensorFlow)
4. **Model Training**: Training the models on the training dataset.
5. **Model Evaluation**: Evaluating the models using metrics like R-squared score.

## Evaluation

The models are evaluated based on their performance on the test dataset. The primary metric used for evaluation is the R-squared score, which indicates the proportion of the variance in the dependent variable that is predictable from the independent variables.

## Results

The results section in the notebook provides a comparison of the performance of different models. The neural network model is trained for multiple epochs to optimize the prediction accuracy.