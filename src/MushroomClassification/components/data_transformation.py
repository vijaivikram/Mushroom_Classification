import os
import sys
import pickle
import numpy as np
import pandas as pd
from MushroomClassification.logger import logging
from MushroomClassification.exception import customexception
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from dataclasses import dataclass
from MushroomClassification.utils.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

        
    
    def get_data_transformation(self):
        
        try:
            logging.info('Data Transformation initiated')

            # Define which columns should be scaled
            categorical_cols = ['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
       'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
       'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
       'stalk-surface-below-ring', 'stalk-color-above-ring',
       'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
       'ring-type', 'spore-print-color', 'population', 'habitat']
            
            
            logging.info('Pipeline Initiated')
            
            

            # Categorigal Pipeline
            cat_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('onehotencoder', OneHotEncoder(sparse_output=False,handle_unknown='ignore')),
                ('scaler',StandardScaler())
                ]
            )
        
            preprocessor = ColumnTransformer([
                
                ('cat_pipeline', cat_pipeline, categorical_cols)
            ])
            
            return preprocessor
            

        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")

            raise customexception(e,sys)
            
    
    def initialize_data_transformation(self, train_path, test_path):
        try:
            logging.info("Reading train and test data from CSV files")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data complete")
            logging.info(f'Train Dataframe Head:\n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head:\n{test_df.head().to_string()}')

            preprocessing_obj = self.get_data_transformation()

            target_column_name = 'class'
            drop_columns = [target_column_name]

            

            

            
            
            logging.info("Extracting features and target columns")
            # Extract features and target columns
            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Read train and test data complete")
            logging.info(f'Train Dataframe Head:\n{input_feature_train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head:\n{input_feature_test_df.head().to_string()}')


            logging.info("Applying preprocessing object on training and testing datasets")
            # Apply preprocessing object
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info("Combining features and target columns into arrays")
            # Combine features and target columns into arrays
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saving preprocessing pickle file")
            # Save preprocessing pickle file
            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path, obj=preprocessing_obj)

            logging.info("Preprocessing pickle file saved")

            return train_arr, test_arr

        except Exception as e:
            logging.error(f"Exception occurred in the initialize_data_transformation: {e}")
            raise customexception(e, sys)