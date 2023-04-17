from resources import constants
import pandas as pd

class BHDUtility:
    def __init__(self):
        pass
    
    def get_dataset(self):
        bhd_df = pd.read_csv(f"{constants.BHD_DATASET}/{constants.BHD_FILE}").drop("address",axis=1)
        value_counts = bhd_df["label"].value_counts()
        selected_labels = list(value_counts[value_counts > 50].keys())
        bhd_df = bhd_df[bhd_df["label"].isin(selected_labels)]
        bhd_df["label"] = bhd_df["label"].astype("category").cat.codes
        return bhd_df