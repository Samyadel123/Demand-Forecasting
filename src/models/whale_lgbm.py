# src/models/whale_lgbm.py
import lightgbm as lgb


class WhaleLGBMForecaster:
    def __init__(self):
        # 'mae' (Mean Absolute Error) ensures the model isn't penalized
        # too heavily by the 4,000,000 max demand outliers.
        self.model = lgb.LGBMRegressor(
            objective="mae",
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            random_state=42,
            n_jobs=-1,  # Uses all available CPU cores
        )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
