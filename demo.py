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
st.set_option('deprecation.showPyplotGlobalUse', False)
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

agree = st.checkbox('I agree to proceed with the above metrics.')

## Load the prediction results
demo_results = pd.read_csv("demo_results.csv")

def predict_label(x):
    if x >= threshold:
        return 1
    if x < threshold:
        return 0



'''
## Any alerts will show here
'''
if agree:
    demo_results["pred_label"] = demo_results["pred_loss"].apply(predict_label)
    attacks = demo_results[demo_results["pred_label"]==1]
    st.write("There are ", attacks.shape[0], "attacks found in the set.")

    if attacks.shape[0] >0:
        for i in range(0, attacks.shape[0]):
            src_ip, des_ip, src_port, des_port, cntype = attacks.iloc[i,0].split('-')
            st.write("Source IP: ", src_ip, " Source Port: ", src_port)
            st.write("Destination IP: ", des_ip, " Destination Port: ", des_port)
            st.write("Time Stamp", attacks.iloc[i,1])



'''
## Display results with labels
#### (Labels shouldn't really be here but it will allow us to get some idea of performance on this small demo set.)
'''
## Display results
hide_dataframe_row_index = """
            <style>
            .row_heading.level0 {display:none}
            .blank {display:none}
            </style>
            """

# Inject CSS with Markdown
st.markdown(hide_dataframe_row_index, unsafe_allow_html=True)

# Display an interactive table
st.dataframe(demo_results)