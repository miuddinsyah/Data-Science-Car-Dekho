# Vehicle Dataset Analysis and Machine Learning Models

This project explores a vehicle dataset and implements machine learning techniques for regression, classification, and clustering. The application can be deployed using two methods: **Flask** (web-based) and **Tkinter** (desktop GUI).

## Project Overview

The project includes:
- Data preprocessing: handling missing values, normalizing features, and encoding categorical variables.
- Exploratory Data Analysis (EDA): heatmaps, histograms, and pair plots.
- Machine learning models:
  - Regression: Linear Regression, Random Forest, Gradient Boosting.
  - Classification: Decision Tree, KNN, SVM.
  - Clustering: K-Means, DBSCAN.
- Performance evaluation with metrics like MSE, accuracy, precision, recall, and silhouette scores.

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

## Deployment Options

### 1. Flask (Web-Based)
- Launches a web application to interact with the models.
- **Steps**:
  1. Clone the repository:
     ```bash
     git clone https://github.com/yourusername/vehicle-dataset-analysis.git
     cd vehicle-dataset-analysis
     ```
  2. Install dependencies:
     ```bash
     pip install -r requirements.txt
     ```
  3. Run the Flask application:
     ```bash
     python app.py
     ```
  4. Open the app in a browser:
     ```
     http://127.0.0.1:5000/
     ```

### 2. Tkinter (Desktop GUI)
- Provides a desktop interface to interact with the models.
- **Steps**:
  1. Clone the repository:
     ```bash
     git clone https://github.com/yourusername/vehicle-dataset-analysis.git
     cd vehicle-dataset-analysis
     ```
  2. Install dependencies:
     ```bash
     pip install -r requirements.txt
     ```
  3. Run the Tkinter application:
     ```bash
     python tkinter_app.py
     ```

  4. Use the graphical interface to perform:
     - Predictions
     - Data visualizations
     - Clustering and classification tasks

## Features

### Flask Application
- **EDA Module**: Visualize dataset, heatmaps, and correlations.
- **Regression**: Predict vehicle selling prices using Linear Regression and SVR.
- **Classification**: Categorize vehicles using KNN and Decision Tree.
- **Clustering**: Group vehicles using K-Means and DBSCAN.

### Tkinter Application
- GUI-based model interaction.
- User-friendly input fields for regression and classification.
- Data visualization support.

## File Structure

- **templates/**: HTML templates for Flask.
- **static/**: Static files (CSS, JS) for Flask.
- **app.py**: Flask web application.
- **tkinter_app.py**: Tkinter-based GUI application.
- **data/**: Dataset files.
- **notebooks/**: Jupyter notebooks for data analysis and model building.

## Authors

- Azmi Taqiuddin Syah
- Kivlan Hakeem Arrouf
- Izzan Muhammad Faiz
