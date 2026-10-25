
# AttritionIQ — Employee Attrition Prediction

A machine learning web application that estimates employee attrition
risk and explores historical workforce trends.

## Features

- Employee attrition dashboard
- Logistic Regression and Random Forest
- Employee-level attrition risk prediction
- Department and overtime analytics
- Job satisfaction analysis
- SQLite prediction history
- CSV export

## Tech Stack

- Python and Flask
- Pandas and NumPy
- Scikit-learn
- Plotly
- SQLite
- HTML, CSS, JavaScript

## Project Setup

1. Create and activate a virtual environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Place the employee dataset at:

   `dataset/employee.csv`

4. Train and evaluate the models:

   ```bash
   python model/train_model.py
   ```

5. Start the Flask application:

   ```bash
   python app/app.py
   ```

6. Open http://127.0.0.1:5000 in your browser.

## Dataset

The CSV must contain an `Attrition` target column with `Yes` and `No`
values. Use an employee dataset with the same feature columns for
training and prediction.

## Responsible Use

Predictions are estimates based on historical patterns, not certain
forecasts. Do not use them as the sole basis for employment decisions.
Protect employee data and review potential bias before real-world use.
