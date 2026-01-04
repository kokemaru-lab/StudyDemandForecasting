from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np

class DemandForecaster:
    """
    需要予測モデルの学習と推論を担当するクラス
    """
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.features = ['price', 'promotion', 'day_of_week', 'month']
        self.target = 'sales'

    def train(self, df: pd.DataFrame):
        """モデルの学習"""
        X = df[self.features]
        y = df[self.target]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model.fit(X_train, y_train)
        
        predictions = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        print(f"Model trained. Mean Absolute Error: {mae:.2f}")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """予測の実行"""
        return self.model.predict(df[self.features])