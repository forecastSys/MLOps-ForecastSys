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

def main():

    st.title("AIDF-CAESARS Forecast System")
    high_level_image = Image.open("../assets/aidf-caesars.png")

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

    company_id = st.text_input("Company ID")
    year_to_predict = int(st.number_input("Year To Predict"))
    y_to_predict = st.text_input("Y to Predict")
    if st.button("Predict"):
        prediction = prediction_service("LGBRegression", company_id, y_to_predict, year_to_predict)

        st.success(
            f"Prediction for {company_id}'s {y_to_predict} in {year_to_predict} is {prediction}"
        )


if __name__ == "__main__":
    main()