
import os
import sys
from src.Exception import CustomException
from src.logging import logging
import pandas as pd 
import numpy
from sklearn.model_selection import train_test_split


from dataclasses import dataclass


#creating rh artifacts of data_ingestion (tarin and test data)

@dataclass
class Data_ingestionconfig:

    train_data_path=os.path.join("artifacts","train.csv")
    test_data_path=os.path.join("artifacts","test.csv")
    raw_data_path=os.path.join("artifacts","raw.csv")



class DataIngestion():

    def __init__(self):
        self.ingestionconfig=Data_ingestionconfig()

    def data_ingestion(self):
        logging.info("Entered the data ingestion part")

        try:

            #reading given data
            df=pd.read_csv("notebooks\data\gemstone.csv")

            #storing raw data as artifacts
            os.makedirs(os.path.dirname(self.ingestionconfig.raw_data_path),exist_ok=True)

            logging.info("the raw data is stored")
            df.to_csv(self.ingestionconfig.raw_data_path,index=False,header=True)
            
            logging.info("train rtest split")
            train_data,test_data=train_test_split(df,test_size=0.30,random_state=42)

            logging.info("the train,test data is saved")
            train_data.to_csv(self.ingestionconfig.train_data_path,index=False,header=True)
            train_data.to_csv(self.ingestionconfig.test_data_path,index=False,header=True)

            logging.info("data ingestio is completed")


            return (
                self.ingestionconfig.train_data_path,
                self.ingestionconfig.test_data_path
            )




        except Exception as e:
            logging.info("Exception in data ingestion")
            raise CustomException(e,sys)




        

