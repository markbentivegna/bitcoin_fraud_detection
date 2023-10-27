from resources import constants
import pandas as pd
import numpy as np

class EthereumTransactionUtility:
    def __init__(self):
        pass
    
    def get_dataset(self):
        column_mappings = {
            "Avg min between sent tnx": "average_time_between_sent",
            "Avg min between received tnx": "average_time_between_received",
            "Time Diff between first and last (Mins)": "time_difference",
            "Sent tnx": "sent_transaction_count",
            "Received Tnx": "received_transaction_count",
            "Number of Created Contracts": "created_contract_count",
            "Unique Received From Addresses": "unique_addresses_received",
            "Unique Sent To Addresses": "unique_addresses_sent",
            "min value received": "min_value_received",
            "max value received ": "max_value_received",
            "avg val received": "average_value_received",
            "min val sent": "min_value_sent",
            "max val sent": "max_value_sent",
            "avg val sent": "average_value_sent",
            "min value sent to contract": "min_ether_sent_to_contract",
            "max val sent to contract": "max_ether_sent_to_contract",
            "avg value sent to contract": "average_ether_sent_to_contract",
            "total transactions (including tnx to create contract": "total_transactions",
            "total Ether sent": "total_ether_sent",
            "total ether received": "total_ether_received",
            "total ether sent contracts": "total_ether_sent_to_contract",
            "total ether balance": "total_ether_balance",
            "min value sent to contract": "min_value_sent_to_contract",
            "max val sent to contract": "max_value_sent_to_contract",
            "FLAG": "label"
        }
        selected_columns = list(column_mappings.values())
        ethereum_df = pd.read_csv(f"{constants.ETHEREUM_DATASET_DIR}/{constants.ETHEREUM_FILE}") 
        ethereum_df = ethereum_df.rename(columns=column_mappings)[selected_columns]
        return ethereum_df