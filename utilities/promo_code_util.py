from resources import constants
import pandas as pd

class PromoCodeUtility:
    def __init__(self):
        pass
    
    def get_dataset(self):
        promo_code_df = pd.read_csv(f"{constants.PROMO_CODE_DATASET}/{constants.PROMO_CODE_FILE}",index_col=0)

        return promo_code_df