import os
import sys
import dill

import numpy as np
import pandas as pd
from src.TravelPackagePrediction.exception import CustomException

from sklearn.metrics import accuracy_score


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train,y_train,X_test,y_test,models):
    try:
         test_acc_report={}
         
         for i in range(len(models)):
             model=list(models.values())[i]
             model.fit(X_train,y_train)

             #Make predictions
             y_test_pred=model.predict(X_test)

             # Evaluate test data
             test_accuracy=accuracy_score(y_test,y_test_pred)

             test_acc_report[model]=test_accuracy


             return test_acc_report
    except Exception as e:
        raise CustomException(e, sys)
   