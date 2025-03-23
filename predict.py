import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import pickle


with open("logistic_regression_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

with open("column_names.pkl", "rb") as f:
    columns = pickle.load(f)


input_df = pd.DataFrame([user_input])

for col in ["Fever", "Cough", "Fatigue", "Difficulty Breathing"]:
    input_df[col] = input_df[col].map({"No": 0, "Yes": 1})

input_df = pd.get_dummies(input_df, columns = ["Gender", "Blood Pressure", "Cholesterol Level"])
# Preprocessing 


input_df = input_df[X_train.columns]


# Gives the probabilities of having a certain disease
probabilities = model.predict_proba(input_df)[0]


# Gives top 3 highest probabilities
top_3 = probabilities.argsort()[-3:][::-1]
top_disease = label_encoder.inverse_transform(top_3)
top_probabilities = probabilities[top_disease]



