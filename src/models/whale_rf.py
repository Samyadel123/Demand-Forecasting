# src/models/whale_rf.py
from sklearn.ensemble import RandomForestRegressor


class RandomForestForecaster:
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=200,  # Number of trees
            max_depth=10,  # Prevent trees from growing too deep and memorizing noise
            min_samples_leaf=5,  # Ensure at least 5 records per leaf to smooth out volatility
            n_jobs=-1,  # Use all available CPU cores
            random_state=42,
        )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
