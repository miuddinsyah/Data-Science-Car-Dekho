# Vehicle Dataset Analysis and Machine Learning Models

This project is part of a final assignment for the Data Science and Analysis course. The project involves exploring a vehicle dataset and applying machine learning techniques such as regression, classification, and clustering to extract insights and make predictions.

## Project Overview

The project covers the following steps:
- Data preprocessing, including handling missing values, normalizing features, and encoding categorical variables.
- Exploratory Data Analysis (EDA) with visualizations like heatmaps, histograms, and pair plots.
- Implementation of regression models (Linear Regression, Random Forest, Gradient Boosting) to predict vehicle selling prices.
- Classification models (Decision Tree, KNN, SVM) to categorize vehicles based on their attributes.
- Clustering (K-Means, DBSCAN) for grouping vehicles into meaningful clusters.
- Performance evaluation using metrics like Mean Squared Error (MSE), accuracy, precision, recall, and silhouette scores.

## Dataset

The dataset contains attributes such as:
- **Name**: Vehicle name (brand and model)
- **Year**: Year of manufacture
- **Selling Price**: Resale price of the vehicle
- **KM Driven**: Kilometers driven
- **Fuel Type**: Diesel, Petrol, etc.
- **Transmission**: Manual or Automatic
- **Engine**: Engine capacity in CC
- **Max Power**: Maximum power in BHP
- **Seats**: Number of seats

### Dataset Size
- **Rows**: 7,253
- **Columns**: 13

## Key Features

### 1. Web Application
- A Flask-based web app is included to interact with various models.
- Available modules:
  - **EDA**: View dataset, heatmaps, and correlation plots.
  - **Linear Regression**: Predict selling prices based on vehicle features.
  - **KNN**: Classify vehicles.
  - **Decision Tree**: Visualize and interpret decision tree classification.
  - **K-Means Clustering**: Group vehicles into clusters.
  - **SVR Regression**: Experiment with support vector regression parameters.

### 2. Visualization
- Heatmaps to analyze feature correlations.
- Histograms for data distribution.
- Cluster plots for K-Means.

### 3. Model Performance
- Regression:
  - **Gradient Boosting**: Best performance with MSE of 7,450,000 and R² = 0.92.
- Classification:
  - **SVM**: Best classification accuracy of 88%.
- Clustering:
  - **DBSCAN**: Higher silhouette score (0.72) compared to K-Means.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/vehicle-dataset-analysis.git
   cd vehicle-dataset-analysis
