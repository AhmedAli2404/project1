import os
import sys
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import pandas as pd
import numpy as np
from src.utils import save_obj
from src.Exception import CustomException
from src.logging import logging


@dataclass
class Data_transformation_config:
    path_of_proceeesor_pickle=os.path.join("artifacts","processor.pkl")


class Data_transformation:

    def __init__(self):
        self.processor_pkl_path=Data_transformation_config()



    def data_transformer_object(self):
        try:
            logging.info("data taransformation started")


            categorical_cols = ['cut', 'color','clarity']
            numerical_cols = ['carat', 'depth','table', 'x', 'y', 'z']
            
            # Define the custom ranking for each ordinal variable

            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']
            
            logging.info('Pipeline Initiated')

            numerical_pipeline=Pipeline(

                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaleer",StandardScaler())
                ]
            )

            categorical_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("ordinalencoder",OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                    ("scaler",StandardScaler())
                ]
            )

            processor_obj=ColumnTransformer([
                ("numerical_pipeline",numerical_pipeline,numerical_cols),
                ("categorical_pipeline",categorical_pipeline,categorical_cols)
            ]
            )

            logging.info("Pipeline completed")


            return processor_obj
        except Exception as e:
            logging.info("Error in crating pipeline processor")
            raise CustomException(e,sys)
        
    def initialising_data_transformation(self,train_path,test_path):

        try:
            
            logging.info("initialising transfomation")
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("the reading of train and test data is completed")

            logging.info("train datta\n{0}".format(train_df.head()))
            logging.info("test data \n{0}".format(test_df.head()))

            logging.info("obtaining preprocessor")
            preprocessing_obj=self.data_transformer_object()

            target_column_name = 'price'
            drop_columns = [target_column_name,'id']
            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column_name]
            
            ## Trnasformating using preprocessor obj
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets.")
            

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_obj(
                filepath=self.processor_pkl_path.path_of_proceeesor_pickle,
                obj=preprocessing_obj
            )


            logging.info("processor pickel is completed")

            return (
                train_arr,
                test_arr,
                self.processor_pkl_path.path_of_proceeesor_pickle
            )

        except Exception as e:
            logging.info("exception in pipeline proceessor")
            raise CustomException(e,sys)





