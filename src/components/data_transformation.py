import os
import sys
from dataclasses import dataclass

from exception import CustomException
from logger import logging
from utils import save_object

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = ['checkout_price', 'base_price', 'op_area', 'discount_amount','discount_percent','weekly_base_price_change',
                                 'weekly_checkout_price_change','4_week_avg_base_price','4_week_avg_checkout_price']
            categorical_columns = []

            # Define the numerical and categorical transformers
            numerical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ])

            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])

            # Combine the transformers into a preprocessor
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_transformer, numerical_columns),
                    ('cat', categorical_transformer, categorical_columns)
                ]
            )

            logging.info("Data transformation pipeline created successfully.")
            return preprocessor

        except Exception as e:
            logging.error(f"Error occurred while creating data transformation pipeline: {e}")
            raise CustomException(e, sys) from e