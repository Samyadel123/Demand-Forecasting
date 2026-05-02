# src/models/ts_prophet.py
from prophet import Prophet
import pandas as pd


class ProphetForecaster:
    def __init__(self):
        # We use '# type: ignore' because the linter incorrectly expects a string.
        # Booleans are the officially supported way to force seasonality in Prophet.
        self.model = Prophet(
            yearly_seasonality=True,  # type: ignore
            weekly_seasonality=True,  # type: ignore
            daily_seasonality=False,  # type: ignore
            changepoint_prior_scale=0.05,
        )

    def train(self, X_train, y_train):
        # Prophet strictly requires columns named 'ds' (datestamp) and 'y' (target)
        df = pd.DataFrame({"ds": pd.to_datetime(X_train["Date"]), "y": y_train})

        # Add your engineered features as extra regressors
        self.regressors = [col for col in X_train.columns if col != "Date"]
        for col in self.regressors:
            self.model.add_regressor(col)
            df[col] = X_train[col].values

        self.model.fit(df)

    def predict(self, X_test):
        # Prepare the future dataframe in Prophet's required format
        future_df = pd.DataFrame({"ds": pd.to_datetime(X_test["Date"])})

        for col in self.regressors:
            future_df[col] = X_test[col].values

        forecast = self.model.predict(future_df)

        # This bypasses the ExtensionArray error safely.
        preds = forecast["yhat"].clip(lower=0).to_numpy()
        return preds
