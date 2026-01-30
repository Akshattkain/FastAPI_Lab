import joblib

def predict_data(X):
    """
    Predict heart disease presence.
    Args:
        X (numpy.ndarray): Input data for which predictions are to be made.
    Returns:
        y_pred (numpy.ndarray): Predicted class labels (0 = healthy, 1 = heart disease).
    """
    model = joblib.load("../model/heart_model.pkl")
    y_pred = model.predict(X)
    return y_pred