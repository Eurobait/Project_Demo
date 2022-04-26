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
fig = PrecisionRecallDisplay.from_predictions(test_summary.Class, test_summary.pred_loss)
st.pyplot(fig)




mold = st.number_input("Chance of mold", 0.01)





st.write("Just Harvest")