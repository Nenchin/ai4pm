import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import warnings
import xgboost as xgb
import matplotlib.pyplot as plt

from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor


st.set_page_config(page_title="AI4PLP", page_icon="ptdf_logo.png",
                   layout="wide")

col1, col2 = st.columns(2)

warnings.filterwarnings("ignore")

# Outlier handler class
class OutlierHandler(BaseEstimator, TransformerMixin):
    def __init__(self, method='iqr', factor=1.5):
        self.method = method
        self.factor = factor

    def fit(self, X, y=None):
        if self.method == 'iqr':
            # Calculate the IQR and bounds
            Q1 = np.percentile(X, 25)
            Q3 = np.percentile(X, 75)
            IQR = Q3 - Q1
            self.lower_bound = Q1 - self.factor * IQR
            self.upper_bound = Q3 + self.factor * IQR
        return self

    def transform(self, X):
        if self.method == 'iqr':
            # Clip outliers to the lower and upper bounds
            X_clipped = np.clip(X, self.lower_bound, self.upper_bound)
            return X_clipped
        else:
            return X
        
col1.image("ptdf_logo.png", width=100)

col1.write("""
           ### Artificial Intelligence for Pipeline Leakage Prediction (AI4PLP)
            """)

#st.markdown("<h2 style='text-align: center; color: black; \
#            '>PTDF Artificial Intelligence for Pipeline Monitoring</h1>", 
#            unsafe_allow_html=True)

data = np.load("sampled_data.npy")
X, y = data[:,:-1].copy(), data[:,-1]

lin_reg_model = joblib.load("lin_reg_model.pkl")
xgb_model = joblib.load("xgb_reg.pkl")
ann_model = joblib.load("ann_model.pkl")

lin_reg_model_pred = lin_reg_model.predict(X)
xgb_model_pred = xgb_model.predict(X)
ann_model_pred = ann_model.predict(X)

avg_pred = np.c_[lin_reg_model_pred, 
                 xgb_model_pred, ann_model_pred].mean(axis=1)
avg_pred_100 = avg_pred[:101]

col2.write("""
               ---
               ---
               ---
               ---
               ---
               """)

progress_bar = st.sidebar.progress(0)
status_text = st.sidebar.empty()
last_rows = pd.DataFrame([0.0], index=[0])
chart = col2.line_chart(last_rows)

for i,j in enumerate(list(avg_pred_100)):
    new_rows = pd.DataFrame([j], index=[i])
    i = round((i/len(avg_pred_100))*100)
    status_text.text("%i%% Complete" % i )
    chart.add_rows(new_rows)
    progress_bar.progress(i)
    last_rows = new_rows
    time.sleep(0.05)

progress_bar.empty()


data = {"l_r": lin_reg_model_pred.tolist(),
        "xgb": xgb_model_pred.tolist(),
        "ann": ann_model_pred.tolist(),
        "avg_pred": avg_pred.tolist()}

pred_df = pd.DataFrame(data)

col1.write("""
           ##### Predictions from L_R, XGB, and ANN
           """)
col1.write(pred_df)
col1.write("---")
