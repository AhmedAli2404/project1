import os
import sys
import pickle
import numpy as np 
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.Exception import CustomException
from src.logging import logging


def save_obj(filepath,obj):
    try:
        k=os.path.dirname(filepath)
        logging.info("creating a pickle file")
        path=os.makedirs(k,exist_ok=True)

        with open(filepath,"wb") as f:
            pickle.dump(obj,f)


        
    except Exception as e:
        logging.info("exception in pickling")
        raise CustomException(e,sys)
    

def evaluate_model(X_train, y_train, X_test, y_test,models):
    logging.info("evalauation assembled")
    try:
        

        result={}

        for i in range(len(models.keys())):

            model=list(models.values())[i]
            model.fit(X_train,y_train)
            y_pred=model.predict(X_test)
            test_model_score = r2_score(y_test,y_pred)

            result[list(models.keys())[i]] =  test_model_score


            return result

    except Exception as e:
        logging.info("error in evaluating model")
        raise Exception(e,sys)
    