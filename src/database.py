import pandas as pd
from sqlalchemy import create_engine, text

class DatabaseManager:
    """
    PostgreSQLとの接続およびデータ操作を管理するクラス
    """
    def __init__(self, db_config: dict):
        """
        db_config: {'user', 'password', 'host', 'port', 'dbname'}
        """
        self.url = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['dbname']}"
        self.engine = create_engine(self.url)

    def save_dataframe(self, df: pd.DataFrame, table_name: str, if_exists: str = 'replace'):
        """DataFrameをデータベースに保存"""
        df.to_sql(table_name, self.engine, if_exists=if_exists, index=False)

    def load_query(self, query: str) -> pd.DataFrame:
        """SQLクエリの結果をDataFrameとして読み込む"""
        return pd.read_sql(query, self.engine)

    def execute_raw_sql(self, sql: str):
        """生のSQLを実行する（テーブル作成など）"""
        with self.engine.connect() as conn:
            conn.execute(text(sql))
            conn.commit()