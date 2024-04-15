import os
import sys
from MushroomClassification.exception import customexception
from MushroomClassification.logger import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from dataclasses import dataclass

@dataclass()
class DataIngestionConfig:
    train_path=os.path.join("artifacts","train.csv")
    test_path=os.path.join("artifacts","test.csv")
    raw_data=os.path.join("artifacts","raw.csv")

class DataIngestion:
    def __init__(self):
        self.get_data_ingestion_config=DataIngestionConfig()

    def Initiate_data_ingestion(self):
        try:
            logging.info("Data ingestion has started")

            df=pd.read_csv(os.path.join("notebooks/data","mushrooms.csv"))

            os.makedirs(os.path.dirname(self.get_data_ingestion_config.raw_data),exist_ok=True)
            df.to_csv(self.get_data_ingestion_config.raw_data,index=False,header=True)

            train_set,test_set=train_test_split(df,test_size=0.20,random_state=42)
            train_set.to_csv(self.get_data_ingestion_config.train_path,index=False,header=True)
            test_set.to_csv(self.get_data_ingestion_config.test_path,index=False,header=True)
            logging.info("files saved to artifacts")

            return (self.get_data_ingestion_config.train_path,self.get_data_ingestion_config.test_path)

        except Exception as e:
            raise customexception(e,sys)