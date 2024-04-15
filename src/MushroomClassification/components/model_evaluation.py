import os
import sys
from sklearn.metrics import accuracy_score,classification_report
import numpy as np
import pickle
from MushroomClassification.utils.utils import load_object


class ModelEvaluation:
    def __init__(self):
        pass

    def eval_metrics(self,actual, pred):
        accuracy = accuracy_score(actual, pred)
        report = classification_report(actual, pred)
        return accuracy, report 

    def initiate_model_evaluation(self,train_array,test_array):
        try:
            X_test,y_test=(test_array[:,:-1], test_array[:,-1])

            model_path=os.path.join("artifacts","model.pkl")
            model=load_object(model_path)

            predicted_qualities = model.predict(X_test)

            (accuracy, report) = self.eval_metrics(y_test, predicted_qualities)
            return accuracy, report 

        except Exception as e:
            raise e