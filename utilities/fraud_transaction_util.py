from resources import constants
import pandas as pd

class FraudTransactionUtility:
    def __init__(self):
        pass
    
    def get_dataset(self):
        fraud_transaction_df = pd.read_csv(f"{constants.FRAUD_TRANSACTION_DATASET}/{constants.FRAUD_TRANSACTION_FILE}",index_col=0)
        fraud_transaction_df["month"] = pd.DatetimeIndex(fraud_transaction_df["TX_DATETIME"]).month
        fraud_transaction_df["day"] = pd.DatetimeIndex(fraud_transaction_df["TX_DATETIME"]).day
        fraud_transaction_df["hour"] = pd.DatetimeIndex(fraud_transaction_df["TX_DATETIME"]).hour
        fraud_transaction_df.drop(["TX_FRAUD", "TX_DATETIME"],axis=1,inplace=True)
        return fraud_transaction_df.rename(columns={"TX_FRAUD_SCENARIO": "label"})