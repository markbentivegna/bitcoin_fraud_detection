from resources import constants
import pandas as pd
import numpy as np

class CreditCardUtility:
    def __init__(self):
        pass
    
    def get_dataset(self):
        creditcard_df = pd.read_csv(f"{constants.CREDITCARD_DATASET}/{constants.CREDITCARD_FILE}") 
        return creditcard_df.rename(columns={"Class": "label"})