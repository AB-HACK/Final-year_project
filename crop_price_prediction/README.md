# Crop Price Prediction for Smallholder Farmers

## Project Overview
This project is a machine learning web application that predicts crop prices for smallholder farmers using historical agricultural market data. It is designed as a BSc Software Engineering final year project and aims to support better decision-making for farm produce sales.

## Folder Structure
```text
crop_price_prediction/
├── config.py
├── data/
│   ├── raw/          ← put your CSV here
│   └── processed/
├── notebooks/
├── src/
│   ├── __init__.py
│   ├── features/
│   │   └── __init__.py
│   ├── models/
│   │   └── __init__.py
│   └── api/
│       └── __init__.py
├── static/
│   ├── css/
│   └── js/
├── templates/
├── tests/
│   └── __init__.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup Instructions
1. Navigate to the project folder:
   ```bash
   cd crop_price_prediction
   ```
2. (Optional) Create and activate a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Add your Kaggle dataset CSV file to:
   ```text
   data/raw/
   ```
5. Run the Flask app (when your app entry file is ready):
   ```bash
   flask run
   ```

## How to Use
1. Open the web application in a browser.
2. Enter crop-related market details in the form (such as crop type, market, date, and other required fields).
3. Submit the form to get a predicted crop price.
4. Use the predicted value to help decide when and where to sell produce.

## Model Information
The project evaluates multiple regression models:
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

The best-performing model is selected using RMSE (Root Mean Squared Error), and the selected model is used for predictions in the web app.

## Data Source
Kaggle public agricultural market dataset (historical crop price data).
