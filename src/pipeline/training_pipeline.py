import sys
import os
from src.logging import logging
from src.Exception import CustomException
from src.components.data_integration import DataIngestion
from src.components.data_tarnsformation import Data_transformation
from src.components.model_trainer import Model_trainer




if __name__=="__main__":
    try:
        ingetion=DataIngestion()
        train_data,test_data=ingetion.data_ingestion()
        transformer=Data_transformation()
        train_array,test_array,_=transformer.initialising_data_transformation(train_data,test_data)
        trainer=Model_trainer()
        trainer.initiate_model_training(train_array,test_array)

        
    except Exception as e:
        logging.info("error in training pipeline")
        raise CustomException(e,sys)