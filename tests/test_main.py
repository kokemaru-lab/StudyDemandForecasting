import pytest
import pandas as pd
from main import DataGenerator, DemandPreprocessor, ForecastModel

@pytest.fixture
def sample_df():
    """テスト用の共通データ"""
    return DataGenerator.generate(20)

def test_data_generator():
    df = DataGenerator.generate(5)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5

def test_preprocessor(sample_df):
    preprocessor = DemandPreprocessor()
    processed_df = preprocessor.transform(sample_df)
    
    # shift(1) を行っているため、データ数は -1 になる
    assert len(processed_df) == len(sample_df) - 1
    assert 'prev_sales' in processed_df.columns
    assert not processed_df.isnull().values.any()

def test_forecast_model_train_and_predict(sample_df):
    preprocessor = DemandPreprocessor()
    cleaned_df = preprocessor.transform(sample_df)
    
    model = ForecastModel()
    model.train(cleaned_df)
    
    prediction = model.predict_next_day(cleaned_df.iloc[-1])
    assert isinstance(prediction, float)
    assert prediction > 0  # 売上が正の値であることを確認