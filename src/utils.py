import os
import sys
import dill

from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file:
            dill.dump(obj, file)
            logging.info(f'Successfully saved object to {file_path}')
    except Exception as e:
        logging.error(f'Error saving object to {file_path}: {e}')
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            train_acc = np.mean(y_train_pred == y_train)
            test_acc = np.mean(y_test_pred == y_test)
            report[list(models.keys())[i]] = {
                'train_accuracy': train_acc,
                'test_accuracy': test_acc
            }
        return report
    except Exception as e:
        raise CustomException(e, sys)