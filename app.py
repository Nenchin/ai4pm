import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import warnings
import matplotlib.pyplot as plt
import tensorflow

from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.compose import ColumnTransformer, make_column_selector
from tensorflow.keras.models import load_model

#TF_ENABLE_ONEDNN_OPTS=0

# Initial page config
st.set_page_config(page_title="RESPECTS", page_icon="ptdf_logo.png",
                   layout="wide", initial_sidebar_state="expanded")

col1, col2 = st.columns(2)

warnings.filterwarnings("ignore")


def logger(x):
    return np.log(x+1)

def make_leak_size_plots(X, model, data_prep):
    if X.shape[1] == 6:
        X_clean = data_prep.transform(X.iloc[:,:4])
        preds = model.predict(X_clean)
        
        plt.plot(X.iloc[:,-2], ".b", label="Target Leak Size")
        plt.plot(preds[:,0], "g+", alpha=0.2, label="Predicted Leak Size")
        plt.grid()
        plt.title("Model Preformance on Leak Size Prediction")
        plt.legend()
        plt.savefig("leak_size.png")
        c2.image("leak_size.png")
    elif X.shape[1] == 4:
        X_clean = data_prep.transform(X.iloc[:,:4])
        preds = model.predict(X_clean)
        plt.plot(preds[:,0], "g+", alpha=0.2, label="Predicted Leak Size")
        plt.grid()
        plt.title("Model Preformance on Leak Size Prediction")
        plt.legend()
        plt.savefig("leak_size.png")
        c2.image("leak_size.png")
        
def make_leak_loc_plots(X, model, data_prep):
    if X.shape[1] == 6:
        X_clean = data_prep.transform(X.iloc[:,:4])
        preds = model.predict(X_clean)
        
        plt.plot(X.iloc[:,-1], ".b", label="Target Leak Size")
        plt.plot(preds[:,1], "g+", alpha=0.2, label="Predicted Leak Size")
        plt.grid()
        plt.title("Model Preformance on Leak Location Prediction")
        plt.legend()
        plt.savefig("leak_loc.png")
        c2.image("leak_loc.png")
    elif X.shape[1] == 4:
        X_clean = data_prep.transform(X.iloc[:,:4])
        preds = model.predict(X_clean)
        plt.plot(preds[:,1], "g+", alpha=0.2, label="Predicted Leak Size")
        plt.grid()
        plt.title("Model Preformance on Leak Location Prediction")
        plt.legend()
        plt.savefig("leak_loc.png")
        c2.image("leak_loc.png")
    

log_transformer = FunctionTransformer(logger)
        

data_preprocessor = joblib.load("data_preprocessing.pkl")


model = load_model("AI4PM_ANN_Model.hdf5")

original_title = '<h1 style="font-family:Arial; text-align: center; color:Green; \
                    font-size: 50px;"> \
                    aRtificial intelligencE baSed Pipeline lEakage prediCtion for downsTream Sector (RESPECTS) \
                    </h1>'
st.markdown(original_title, unsafe_allow_html=True)

st.sidebar.header("Import custom data (eg., features.csv )")

col1, col2, col3 = st.columns([1,4,1])   

col1.image("ptdf_logo.png", width=100)
col2.header("PTDF/2022/FUTMINNA/RESPECTS")
col3.image("AEIRG-removebg-preview.png", width=120)

st.write("---")


features = st.sidebar.file_uploader("Upload file", type={"csv"})

c1, c2 = st.columns(2)

if features is not None:
    upload_features = None
    upload_features = pd.read_csv(features)
    st.sidebar.write(upload_features)
    if st.sidebar.button("Predict Uploaded Data"):
        upload_features_cleaned = data_preprocessor.transform(upload_features)
        prediction = model.predict(upload_features_cleaned)
        st.sidebar.write("# Predicted leak sizes and locations")
        prediction_df = pd.DataFrame(prediction,
                                     columns=["leak_size_p", "leak_location_p"])
        st.sidebar.write(pd.concat([upload_features, prediction_df],
                                   axis=1))
        c2.write(make_leak_loc_plots(upload_features, model=model, data_prep=data_preprocessor))
        c2.write(make_leak_size_plots(upload_features, model=model, data_prep=data_preprocessor))
        



c2.write("MSE score")
c2.write(pd.DataFrame(np.array([[0.31341,0.24198]]),
                      columns=["leak_size", "leak_location"], index=[0]))

def input_features():
    pressure_in = c1.number_input("Inlet Pressure", value=None,
                                  placeholder="Type the inlet pressure")
    
    flow_in = c1.number_input("Inlet Flowrate", value=None,
                              placeholder="Type the inlet flowrate")
    
    pressure_out = c1.number_input("Outlet Pressure", value=None,
                                  placeholder="Type in the inlet pressure")
    
    flow_out = c1.number_input("Outlet Flowrate", value=None,
                              placeholder="Type the outlet flowrate")
    
    data = {
            "pressure_in": pressure_in,
            "flow_in": flow_in,
            "pressure_out": pressure_out,
            "flow_out": flow_out
            }
    features_df = pd.DataFrame(data, index=[0])
    
    c1.write(features_df)
    
    return features_df

input_df = input_features()

if c1.button("Predict Input Data"):
    input_data_cleaned = data_preprocessor.transform(input_df)
    prediction = model.predict(input_data_cleaned)
    c1.write(" # Predicted leak size and location")
    c1.write(pd.DataFrame(prediction,
                          columns=["leak_size", "leak_location"],
                          index=[0]))
    c1.write("---")

