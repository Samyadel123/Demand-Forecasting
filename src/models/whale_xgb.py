# src/models/whale_xgb.py
import xgboost as xgb


class WhaleForecaster:
    def __init__(self):
        # Huber loss or MAE is critical here to prevent the
        # massive outliers from hijacking the gradient
        self.model = xgb.XGBRegressor(
            objective="reg:absoluteerror",
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
        )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
