from cmath import nan
import streamlit as st
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import joblib

torch.manual_seed(87)
np.random.seed(87)


## Load Threshold
test_summary = pd.read_csv("test_results.csv")


'''
## Display precision recall curve
'''
## Display precision recall curve

PrecisionRecallDisplay.from_predictions(test_summary.Class, test_summary.pred_loss)
st.pyplot()


'''
## Enter your performance preference
'''
## Collect user preference
u_precision = st.number_input("Expected Precision", 0.00)
u_recall = st.number_input("Expected Recall", 0.00)




# Search for threshold based on precision
def threshold_by_precision(p):
    # we need to exclude first 1000 rows in the search process
    df = test_summary.copy()
    df = df.iloc[1000:,:]
    df["Precision"] = abs(df["Precision"] - u_precision)
    df = df.sort_values(by=['Precision'], ascending=True)
    threshold, type, precision, recall = test_summary.iloc[df.index[0], :]
    return threshold, precision, recall


# Search for threshold based on precision
def threshold_by_recall(p):
    # we need to exclude first 1000 rows in the search process
    df = test_summary.copy()
    df = df.iloc[1000:,:]
    df["Recall"] = abs(df["Recall"] - u_recall)
    df = df.sort_values(by=['Recall'], ascending=True)
    threshold, type, precision, recall = test_summary.iloc[df.index[0], :]
    return threshold, precision, recall


if u_precision == 0.00 and u_recall != 0.00:
    threshold, precision, recall = threshold_by_recall(u_recall)
elif u_precision != 0.00 and u_recall == 0.00:
    threshold, precision, recall = threshold_by_precision(u_precision)
else: 
    st.write("Input is not valid. Only one metric can be non-zero to initiate a search" )
    threshold, precision, recall = nan, nan, nan


'''
## Expected performance summary
'''

# output the expected values
st.write("The current threshold is",  threshold)
st.write("The current expected precision is",  precision)
st.write("The current expected recall is",  recall)


## Load the prediction results







# Show the prediction results on screen