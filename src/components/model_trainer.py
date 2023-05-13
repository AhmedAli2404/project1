import os
import sys
from src.logging import logging
from src.Exception import CustomException
from src.utils import save_obj,evaluate_model

from sklearn.linear_model import LinearRegression, Ridge,Lasso,ElasticNet
from dataclasses import dataclass
import numpy as np
import pandas as  pd


@dataclass
class Model_train_config:
    model_train_path=os.path.join("artifacts","model.pkl")


class Model_trainer:
    def __init__(self):
        self.model_path=Model_train_config()

    

    def initiate_model_training(self,train_array,test_array):

        try:
            logging.info("initiated the process of model Training")

            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )


            models={
            'LinearRegression':LinearRegression(),
            'Lasso':Lasso(),
            'Ridge':Ridge(),
            'Elasticnet':ElasticNet()
        }
            

            modeltrain_result=evaluate_model(X_train, y_train, X_test, y_test,models)


            perfrct_r2=max(modeltrain_result.values())

            perfect_algo=list(modeltrain_result.keys())[
                list(modeltrain_result.values()).index(perfrct_r2)
            ]


            perfect_model=models[perfect_algo]
            print('\n====================================================================================\n')
            print(f'Best Model Found , Model Name : {perfect_model} , R2 Score : {perfrct_r2}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {perfect_model} , R2 Score : {perfrct_r2}')


            save_obj(
                filepath=self.model_path.model_train_path,
                obj=perfect_model
            )

        except Exception as e:
            logging.info("error in training a model")
            raise CustomException(e,sys)