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


'''
# Load Model
class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.Tanh(),
            nn.Linear(encoding_dim, hidden_dim),
            nn.LeakyReLU(),
            )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim),
            nn.LeakyReLU()
            )
    def forward(self , x):
        x = self.encoder(x)
        x = self.decoder(x)
        return(x)

device = torch.device("cpu")
scaler = joblib.load('scaler.gz')
PATH = "current_model.pt"
model = torch.load(PATH)
model.eval()
'''


# Load Threshold
test_summary = pd.read_csv("test_results.csv")

## Display precision recall curve
'''
PrecisionRecallDisplay.from_predictions(test_summary.Class, test_summary.pred_loss)
plt.ylim((0,1))
plt.show()
'''



# Collect user preference
u_precision = st.number_input("Expected Precision", 0.00)
u_recall = st.number_input("Expected Recall", 0.00)


# Add a line to verify that only one of the input precision / recall is not 0



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



# output the expected values
st.write("The current threshold is",  threshold)
st.write("The current expected precision is",  precision)
st.write("The current expected recall is",  recall)



'''
# Prediction
# read a small dataset for demo, there should not be any label
# Label is included here for revision reference only
batch_test = pd.read_csv("demo_test.csv") 
batch_id = batch_test["Flow_ID"]
batch_timestamp = batch_test["Time_stamp"]

batch_X = batch_test.iloc[:,2:86]

batch_loader = DataLoader(batch_X, batch_size = 2048, shuffle=False)


criterion = nn.MSELoss(reduction='none')

with torch.no_grad():
        model.eval() 
        anomaly_scores = []

        for X in batch_loader:
            X = X.to(device)
            outputs = model(X)
            loss = criterion(outputs, X).detach().cpu().numpy()
            anomaly_scores.append(loss)

            del X
            del outputs
            del loss

anomaly_scores = np.sum(np.concatenate(anomaly_scores, axis=0), axis=1)

errorDF = pd.DataFrame(anomaly_scores, columns=["pred_loss"])
errorDF['batch_id'] = batch_id
errorDF['Time_stamp'] = batch_timestamp
'''