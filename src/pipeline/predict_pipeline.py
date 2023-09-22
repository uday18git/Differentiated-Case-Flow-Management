import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    def predict(self,features):
        try:
            model_path = "artifacts\model.pkl"
            preprocessor_path = "artifacts\preprocessor.pkl"
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
                 People_affected: int,
                 Compensation: int,
                 Time_period: int,
                 Death: int):
        self.People_affected = People_affected
        self.Compensation = Compensation
        self.Time_period = Time_period
        self.Death = Death

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "People affected": [self.People_affected/450000000],
                "Compensation": [self.Compensation/120000000],
                "Time period": [self.Time_period/29],
                "Death": [self.Death/52]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)
