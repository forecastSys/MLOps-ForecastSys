import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from src.mlops.pipelines import prediction_service
import json
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import mlflow
from zenml.client import Client

def main():

    abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../assets'))
    # high_level_image = Image.open(os.path.join(abs_path, 'aidf-caesars.png'))
    # st.image(high_level_image, caption="ForecastSys Pipeline", use_column_width=True)
    st.title("AIDF-CAESARS: Forecast System v0.0.1")
    st.markdown(
    """ 
    #### Problem Statement 
     The objective here is to assist the AIDF-Caesars Report Generation system by building a production-ready pipeline using ZenML to forecast several financial indicators for companies. These indicators include:

     - SALES_REV_TURN: Sales Revenue or Turnover
     - CF_CASH_FROM_OPER: Cash Flow from Operations
     - ARD_CAPITAL_EXPENDITURES: Capital Expenditures
     - EBITDA: Earnings Before Interest, Taxes, Depreciation, and Amortization
     This pipeline will help automate financial forecasting and provide valuable insights for report generation.
     """)
    st.markdown(
    """ 
    Below is a figure of the whole pipeline.
    """
    )
    whole_pipeline_image = Image.open(os.path.join(abs_path, 'mlops-workflow.png'))
    st.image(whole_pipeline_image, caption="ForecastSys Workflow")

    company_id = st.text_input("Company ID")
    year_to_predict = int(st.number_input("Year To Predict"))
    y_to_predict = st.text_input("Y to Predict")

    bbunique2u3 = pd.read_csv(os.path.join(abs_path, 'bbunique2u3.csv'))
    company_name = bbunique2u3[bbunique2u3.ID_BB_UNIQUE == company_id]['Company_name'].iloc[0]
    print(company_name)
    if st.button("Predict"):
        if year_to_predict == 2026:
            prediction = prediction_service("LGBRegression", company_id, y_to_predict, year_to_predict)
            # run_id = run_metadata.id
            # client = Client()
            # pipeline_run = client.get_pipeline_run(f"prediction_service-{run_id}")
            # predict_step = pipeline_run.get_step("predict")
            st.success(
                f"Prediction for {company_name}({company_id})'s {y_to_predict} in {year_to_predict} is {prediction}."
            )
        else:
            st.success(
                f"Prediction only available for 2026."
            )


if __name__ == "__main__":
    main()