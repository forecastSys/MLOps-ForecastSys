from src.mlops.abstractions import DataReconstructionStrategyABC
from src.mlops.model import ARModel
from src.mlops.model import ModelCaller
from typing import Union, Tuple, List
import pandas as pd
import numpy as np
from tqdm import tqdm

class BBGDataReconstructionTSHelper:

    @staticmethod
    def data_reconstruction_helper(df, target_column, model):
        """
        Processes a DataFrame containing time series data, segments it into individual series,
        and applies a forecasting model to each series.

        Parameters:
        - df (DataFrame): The input DataFrame containing the time series data.
        - target_column (str): The name of the target column in the DataFrame to forecast.
        - model (function): A forecasting function that takes in training_pipeline data and the number of points to forecast.

        Returns:
        - result (list): A list of forecasted values for each series.
        - result_flatten (list): A flattened list of all forecasted values.
        - training_data_with_forecast_result (list): A combined list of training_pipeline data and forecasted values for all series.
        """
        training_data = []
        forecasting_point = 0
        result = []
        training_data_with_forecast_result = []

        for index, row in df.iterrows():
            if pd.notna(row[target_column]):
                training_data.append(row[target_column])
            else:
                forecasting_point += 1

        if len(training_data) > 3:
            # For the final timeseries
            forecast_values = ModelCaller(strategy=model).train(training_points=training_data, forecast_points=forecasting_point)
            result.append(forecast_values)
            training_data_with_forecast_result.append(training_data)
            training_data_with_forecast_result.append(forecast_values)

            result_flatten = [item for sublist in result for item in sublist]
            training_data_with_forecast_result = [item for sublist in training_data_with_forecast_result for item in sublist]
        else:
            forecast_values = [training_data[-1] for _ in range(forecasting_point)]
            result.append(forecast_values)
            training_data_with_forecast_result.append(training_data)
            training_data_with_forecast_result.append(forecast_values)
        result_flatten = [item for sublist in result for item in sublist]
        return result, result_flatten, training_data_with_forecast_result



class BBGDataReconstructionTS(DataReconstructionStrategyABC):

    def handle_data(self, companyID_df_dict: dict, model: Union[ARModel]) -> dict:

        self.logger.info(f"start data reconstruction - model: {model.__class__.__name__}")
        companyID_df_postConstru_dict = {}
        for id_bb_unique, df_dict in tqdm(companyID_df_dict.items(),
                               desc=f"running data reconstruction - calling: {BBGDataReconstructionTS().__class__.__name__} - model: {model.__class__.__name__}"):
            """
            df_dict with key - value: {
                'df_preConstru': sub_df,
                'df_train': sub_df_train,
                'df_test': df_test,
                'nan_rows': nan_rows,
            }
            """
            forecast_results = {}
            df_x = df_dict["df_preconstru"][self.x_cols_to_process]
            for col in self.x_cols_to_process:
                if col in df_x.columns:
                    result, result_flatten, training_data_with_forecast_result = \
                        BBGDataReconstructionTSHelper.data_reconstruction_helper(df_x, col, model)
                    forecast_results[col] = training_data_with_forecast_result
            df_forecast_x = pd.DataFrame(forecast_results)
            temp_dict = {
                'df_preconstru': df_dict['df_preconstru'],
                'df_train': df_dict['df_train'],
                'df_test': df_dict['df_test'],
                'nan_rows': df_dict['nan_rows'],
                f'df_x_{model.__class__.__name__}_postConstru'.lower().replace('model', ''): df_forecast_x
            }
            companyID_df_postConstru_dict[id_bb_unique] = temp_dict

        return companyID_df_postConstru_dict