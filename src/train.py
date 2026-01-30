from xgboost import XGBClassifier
import joblib
from data import load_data, split_data

def fit_model(X_train, y_train):
    """
    Training an XGBoost Classifier.
    """
    xgb_classifier = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    xgb_classifier.fit(X_train, y_train)
    joblib.dump(xgb_classifier, "../model/heart_model.pkl")

if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    fit_model(X_train, y_train)
    print("Model trained and saved to ../model/heart_model.pkl")