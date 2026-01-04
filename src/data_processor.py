import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class DataProcessor:
    """
    データの生成とクレンジングを担当するクラス
    """
    def generate_dummy_data(self, n_days: int = 365) -> pd.DataFrame:
        """
        需要予測用のダミーデータを生成する
        """
        dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(n_days)]
        
        data = {
            'date': dates,
            'store_id': [1] * n_days,
            'product_id': [101] * n_days,
            'price': np.random.uniform(100, 200, n_days).astype(float),
            'promotion': np.random.choice([0, 1], n_days, p=[0.8, 0.2]).astype(int),
            'sales': np.random.poisson(20, n_days).astype(float) # 計算用にfloatで生成
        }
        
        df = pd.DataFrame(data)
        
        # 意図的に欠損値や外れ値を混ぜる
        df.loc[10:15, 'sales'] = np.nan
        df.loc[20, 'sales'] = 5000.0  # 外れ値
        
        return df

    def cleanse_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        データのクレンジングを行う
        """
        df = df.copy()
        
        # 計算エラーを防ぐためsalesをfloat型に強制
        df['sales'] = df['sales'].astype(float)
        
        # 1. 欠損値補完（中央値）
        if df['sales'].isnull().any():
            df['sales'] = df['sales'].fillna(df['sales'].median())
        
        # 2. 外れ値のクリッピング
        # データが少ない場合に備え、標準偏差が0でないことを確認
        std = df['sales'].std()
        if not pd.isna(std) and std > 0:
            upper_limit = df['sales'].mean() + 3 * std
            df.loc[df['sales'] > upper_limit, 'sales'] = upper_limit
        
        # 3. 日付情報の分解
        df['date'] = pd.to_datetime(df['date'])
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        
        return df