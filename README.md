# Propensify Capstone Project

This project focuses on building a machine learning model using the **Propensify** dataset to predict customer responses to marketing campaigns.

## Table of Contents

- [Project Overview](#project-overview)
- [Data Description](#data-description)
- [Modeling Steps](#modeling-steps)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Deployment on SageMaker](#deployment-on-sagemaker)
- [Contributors](#contributors)

## Project Overview

The goal of this project is to analyze customer data, handle missing values, perform feature engineering, and build a predictive model to identify which customers are more likely to respond positively to a marketing campaign.

## Data Description

The dataset includes various customer features such as:

- **custAge**: Age of the customer
- **profession**: Type of job
- **marital**: Marital status
- **schooling**: Education level
- **loan**: Status of personal loans
- **responded**: Target variable indicating whether the customer responded to the campaign

The dataset is split into two part Historical (train.xlsx) and unseen data (test.xlsx) which can be found in data folder.

## Modeling Steps

1. **Data Import and EDA**: Importing data and performing exploratory data analysis (EDA).
2. **Handling Missing Values**: Imputing missing data using various techniques.
3. **Feature Engineering**: Encoding categorical variables and transforming features for better model performance.
4. **Modeling**: Building machine learning models, including Random Forest and XGBoost classifiers.
5. **Evaluation**: Evaluating model performance with accuracy, recall, and precision.

## Technologies Used

- **Python**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **XGBoost**
- **Seaborn** & **Matplotlib** for data visualization

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Rahulyadav4251/Propensify-Capstone-Project.git


## Usage

After installing the dependencies, you can run the Jupyter notebooks to:

- Perform data preprocessing
- Train the machine learning model
- Evaluate the model performance


## Deployment on SageMaker

 Using the scripts in the src folder, the model is deployed to Amazon SageMaker for serverless inference, where it can be invoked for predictions via a real-time endpoint.To deploy the trained model on Amazon SageMaker for serverless inference, the necessary steps are detailed in the src folder of this repository.

