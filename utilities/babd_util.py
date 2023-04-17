from resources import constants
import pandas as pd
import numpy as np

class BABDUtility:
    def __init__(self):
        pass
    
    def get_dataset(self):
        babd_df = pd.read_csv(f"{constants.BABD_DATASET}/{constants.BABD_FILE}").drop("account", axis=1)
        babd_df['SW'] = np.where(babd_df['SW'] == 'SA', 0, 1)
        return babd_df