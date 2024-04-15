import os
import sys
import pickle
import numpy as np
import pandas as pd
from MushroomClassification.logger import logging
from MushroomClassification.exception import customexception
from dataclasses import dataclass
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from MushroomClassification.utils.utils import save_object
from MushroomClassification.utils.utils import evaluate_model


@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initate_model_training(self,train_array,test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            # Extracting the first 5 rows for logging
            logging.info(f'X_train (first 5 rows):\n{pd.DataFrame(train_array[:,:-1]).head()}')
            logging.info(f'y_train (first 5 rows):\n{pd.DataFrame(train_array[:,-1]).head()}')
            logging.info(f'X_test (first 5 rows):\n{pd.DataFrame(test_array[:,:-1]).head()}')
            logging.info(f'y_test (first 5 rows):\n{pd.DataFrame(test_array[:,-1]).head()}')
            

            models={
                    'LogisticRegression':LogisticRegression(),
                    'DecisionTreeRegressor':DecisionTreeClassifier(),
                    'RandomForestRegressor':RandomForestClassifier(),
                    'KNN': KNeighborsClassifier(),
                    'SVC': SVC(),
                    'GaussianNB': GaussianNB()

                 }
            
            logging.info('Evaluating models...')
            model_report:dict=evaluate_model(X_train,y_train,X_test,y_test,models)
            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')

            # To get best model score from dictionary 
            best_model_score = max(model_report.values(), key=lambda x: x['test_score'])


            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]

            print(f'Best Model Found , Model Name : {best_model_name} , Accuracy Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , Accuracy Score : {best_model_score}')

            save_object(
                 file_path=self.model_trainer_config.trained_model_file_path,
                 obj=best_model
            )
          

        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise customexception(e,sys)