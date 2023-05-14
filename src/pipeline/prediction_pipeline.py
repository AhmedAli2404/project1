import os
import sys
from src.logging import logging
from src.Exception import CustomException
from src.utils import load_pickel
import pandas as pd

class Prediction_pipeline:

    def __init__(self):
        pass

    def get_predicted_value(self,features):
        try :
            processror_path=os.path.join("artifacts","processor.pkl")
            model_path=os.path.join("artifacts","model.pkl")


            processor=load_pickel(processror_path)
            model=load_pickel(model_path)

            scaled_data=processor.transform(features)

            pred=model.predict(scaled_data)

            return pred


        except Exception as e:
            logging.info("the error in unpickling")
            raise CustomException(e,sys)
        


class CustomData:
    def __init__(self,
                 carat:float,
                 depth:float,
                 table:float,
                 x:float,
                 y:float,
                 z:float,
                 cut:str,
                 color:str,
                 clarity:str):
        
        self.carat=carat
        self.depth=depth
        self.table=table
        self.x=x
        self.y=y
        self.z=z
        self.cut = cut
        self.color = color
        self.clarity = clarity

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'carat':[self.carat],
                'depth':[self.depth],
                'table':[self.table],
                'x':[self.x],
                'y':[self.y],
                'z':[self.z],
                'cut':[self.cut],
                'color':[self.color],
                'clarity':[self.clarity]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e,sys)
