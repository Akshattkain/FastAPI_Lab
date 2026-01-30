# Heart Disease Prediction API

## Overview

A FastAPI application that predicts heart disease using an XGBoost classifier.

**Changes from original lab:**
- Dataset: Heart Disease (UCI) instead of Iris
- Model: XGBoost instead of Decision Tree

## Setup

1. Create and activate virtual environment:
    ```bash
    python -m venv lab1_env
    source lab1_env/bin/activate
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Lab

1. Train the model:
    ```bash
    cd src
    python train.py
    ```

2. Start the API:
    ```bash
    uvicorn main:app --reload
    ```

3. Test at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## Sample Request

```json
{
    "age": 55,
    "sex": 1,
    "cp": 2,
    "trestbps": 140,
    "chol": 250,
    "fbs": 0,
    "restecg": 1,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 1.5,
    "slope": 1,
    "ca": 0,
    "thal": 3
}
```

## Project Structure

```
FastAPI_Labs/
├── model/
│   └── heart_model.pkl
├── src/
│   ├── data.py
│   ├── main.py
│   ├── predict.py
│   └── train.py
├── heart.csv
├── README.md
└── requirements.txt
```
