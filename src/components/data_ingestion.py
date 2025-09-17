import os
import sys
from exception import CustomException
from logger import logging
import pandas as pd
from dataclasses import dataclass 
from sklearn.model_selection import train_test_split


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    merged_data_path: str = os.path.join('artifacts', 'merged.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Enter data ingestion stage")

        try:

            weekly_demand = pd.read_csv('notebook/data/train.csv')
            center_info = pd.read_csv('notebook/data/fulfilment_center_info.csv') 
            meal_info = pd.read_csv('notebook/data/meal_info.csv')
            test = pd.read_csv('notebook/data/test.csv')
            logging.info("Read the dataset as dataframe")

            # Adding a placeholder column for predictions in the test dataset
            test['num_orders'] = 0
            
            logging.info("Merging all the datasets into a single dataset")
            # merging all the data sets into single datset for analysis
            data = pd.concat([weekly_demand, test], axis=0)
            data = data.merge(center_info, on='center_id', how='left')
            data = data.merge(meal_info, on='meal_id', how='left')


            # logging.info(f"test data: {test.head(5)}")
            # logging.info(f"merged data: {data.head(5)}")
            # logging.info(f"merged data shape: {data.shape}")
            # logging.info(f"merged data columns: {data.columns}")

            os.makedirs(os.path.dirname(self.ingestion_config.merged_data_path), exist_ok=True)
            data.to_csv(self.ingestion_config.merged_data_path, index=False)

            logging.info("Train test split initiated")
            train_set = data[data['num_orders'] != 0]
            test_set = data[data['num_orders'] == 0]

            logging.info(f"train set: {train_set.head(5)}")
            logging.info(f"Train set shape: {train_set.shape}")
            logging.info(f"test set: {test_set.head(5)}")
            logging.info(f"Test set shape: {test_set.shape}")


            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)

            logging.info("Data Ingestion completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()