# src/models/baseline_lr.py
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import numpy as np


class LinearForecaster:
    def __init__(self):
        # We use Ridge (Linear Regression with L2 Regularization)
        # to handle highly correlated engineered features (like lags and rolling means)
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=1.0, solver="auto")

    def train(self, X_train, y_train):
        # Linear models cannot handle massive outliers well.
        # Applying a log1p transform to the target limits the impact of the 4M spikes.
        self.y_train_log = np.log1p(y_train)

        # Scale the engineered features
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, self.y_train_log)

    def predict(self, X_test):
        X_test_scaled = self.scaler.transform(X_test)
        preds_log = self.model.predict(X_test_scaled)

        # Reverse the log transform to return to the original demand scale
        # Use expm1 to accurately reverse log1p, and clip to 0 so we don't predict negative demand
        preds = np.expm1(preds_log)
        return np.clip(preds, a_min=0, a_max=None)
