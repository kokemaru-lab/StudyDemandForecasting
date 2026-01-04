from src.database import DatabaseManager
from src.data_processor import DataProcessor
from src.model import DemandForecaster

def main():
    # 1. 設定 (実際は環境変数やconfigファイルから読み込む)
    db_config = {
        'user': 'postgres',
        'password': 'Dev_1103_yarimasu',
        'host': 'localhost',
        'port': '5432',
        'dbname': 'demand_db'
    }

    # 2. クラスの初期化
    # db_manager = DatabaseManager(db_config) # DB環境がある場合に使用
    processor = DataProcessor()
    forecaster = DemandForecaster()

    print("--- 1. Generating Data ---")
    raw_df = processor.generate_dummy_data()
    
    print("--- 2. Cleansing Data ---")
    clean_df = processor.cleanse_data(raw_df)

    print("--- 3. Training Model ---")
    forecaster.train(clean_df)

    print("--- 4. Generating Predictions ---")
    # 未来のダミー特徴量を作成して予測
    future_data = clean_df.tail(7).copy() # 直近7日分をテストとして予測
    predictions = forecaster.predict(future_data)
    future_data['predicted_sales'] = predictions
    
    print("\nForecast for the next 7 periods:")
    print(future_data[['date', 'sales', 'predicted_sales']])

    # 5. DB保存例 (コメントアウト)
    db_manager.save_dataframe(clean_df, 'cleaned_sales_history')
    print("\nData saved to PostgreSQL.")

if __name__ == "__main__":
    main()