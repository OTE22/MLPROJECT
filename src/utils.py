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