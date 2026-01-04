import pytest
import pandas as pd
import numpy as np
from src.data_processor import DataProcessor
from src.model import DemandForecaster

@pytest.fixture
def sample_data():
    processor = DataProcessor()
    return processor.generate_dummy_data(n_days=100)

def test_data_generation(sample_data):
    """データ生成が正しく行われているか"""
    assert len(sample_data) == 100
    assert 'sales' in sample_data.columns
    assert 'date' in sample_data.columns

def test_data_cleansing(sample_data):
    """クレンジングによって欠損値がなくなっているか"""
    processor = DataProcessor()
    cleansed_df = processor.cleanse_data(sample_data)
    
    assert cleansed_df['sales'].isnull().sum() == 0
    assert 'day_of_week' in cleansed_df.columns
    assert 'month' in cleansed_df.columns

def test_model_training_and_prediction(sample_data):
    """モデルが学習・予測可能か"""
    processor = DataProcessor()
    df = processor.cleanse_data(sample_data)
    
    forecaster = DemandForecaster()
    forecaster.train(df)
    
    predictions = forecaster.predict(df.head(10))
    assert len(predictions) == 10
    assert isinstance(predictions, np.ndarray)

def test_outlier_clipping():
    """外れ値の処理が機能しているか（データ量を増やしてテスト）"""
    processor = DataProcessor()
    # 大量の正常データの中に1つだけ異常値を混ぜる
    normal_sales = [10.0] * 50
    outlier_sales = [1000.0]
    
    df = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=51),
        'sales': normal_sales + outlier_sales
    })
    
    cleansed = processor.cleanse_data(df)
    # 51件あれば3σ法が正しく機能し、1000は削られるはず
    assert cleansed['sales'].max() < 1000