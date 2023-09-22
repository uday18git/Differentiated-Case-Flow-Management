import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer
@dataclass

class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts','train.csv')
    test_data_path: str=os.path.join('artifacts','test.csv')
    raw_data_path: str=os.path.join('artifacts','data.csv')
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_excel('notebook\data\crime.xlsx')
            df = df.dropna()
            weights = {
                'People affected': 0.25,
                'Compensation': 0.25,
                'Time period': 0.25,
                'Death': 0.25
            }
            numeric_columns = ['People affected', 'Compensation', 'Time period', 'Death']
            df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
            df[numeric_columns] = df[numeric_columns].apply(lambda x: pd.to_numeric(x, errors='coerce'))
            for column in numeric_columns:
                df[column] = df[column] / df[column].max()

            # Calculate the "Priority Score" for each row
            df['Priority Score'] = df.apply(lambda row: sum(row[column] * weights[column] for column in weights),
                                            axis=1)

            # Rescale "Priority Score" to the range of 0 to 10
            min_score = df['Priority Score'].min()
            max_score = df['Priority Score'].max()
            df['Priority Score'] = 0 + (df['Priority Score'] - min_score) * (10 - 0) / (max_score - min_score)

            logging.info("Read the dataset as dataframe")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info("Train test split initiated")
            train_set,test_set = train_test_split(df,test_size=0.1,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("Ingestion of data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr,test_arr,_ = data_transformation.initiate_data_transformation(train_data, test_data)

    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr, test_arr))
