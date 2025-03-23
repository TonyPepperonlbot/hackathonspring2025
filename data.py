#!/usr/bin/env python
# coding: utf-8



# In[3]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
import pickle


# In[4]:


df = pd.read_csv("Disease_symptom_and_patient_profile_dataset.csv")


# In[5]:


df.head(3)


# In[6]:


df = df[df["Disease"].map(df["Disease"].value_counts()) > 4]


# In[7]:


df = df.rename(columns = {"Difficulty Breathing": "Difficulty_Breathing", "Blood Pressure": "Blood_Pressure", "Cholesterol Level": "Cholesterol_Level"})


# In[9]:


df


# In[11]:


df['Outcome Variable'] = df['Outcome Variable'].map({'Negative': 0, 'Positive': 1})
df['Cholesterol_Level'] = df['Cholesterol_Level'].map({'Low': 0, 'Normal': 1, 'High': 2})
df['Blood_Pressure'] = df['Blood_Pressure'].map({'Low': 0, 'Normal': 1, 'High': 2})
df["Fever"] = df["Fever"].map({'No': 0, 'Yes': 1})
df["Cough"] = df["Cough"].map({'No': 0, 'Yes': 1})
df["Fatigue"] = df["Fatigue"].map({'No': 0, 'Yes': 1})
df["Difficulty_Breathing"] = df["Difficulty_Breathing"].map({'No': 0, 'Yes': 1})
df["Gender"] = df["Gender"].map({'Female': 0, 'Male': 1})


df


# In[12]:


sns.heatmap(df.drop("Disease", axis=1).corr(), annot=True, cmap="coolwarm")
plt.show()


# In[13]:


label_encoder = LabelEncoder()
disease_encoder = label_encoder.fit_transform(df["Disease"])


# In[15]:


X = df[["Fever","Cough","Fatigue","Difficulty_Breathing","Blood_Pressure","Cholesterol_Level"]]
y = df["Disease"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[16]:


ros = RandomOverSampler(random_state=42)

X_resampled, y_resampled = ros.fit_resample(X_train, y_train)


# In[17]:


model = LogisticRegression()
model.fit(X_resampled, y_resampled)
#df.groupby(by='Disease').mean()


# In[18]:


y_pred = model.predict(X_resampled)
y_pred


# In[19]:


y_pred = model.predict(X_resampled)  # Use only features to predict labels

# Compute accuracy with true labels
accuracy = accuracy_score(y_resampled, y_pred)


# In[20]:


accuracy


# In[21]:


print(pd.Series(y_resampled.value_counts()))



# In[22]:


with open("logistic_regression_model.pkl", "wb") as f:
    pickle.dump(model, f)


# In[23]:


with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)


# In[24]:


with open("column_names.pkl", "wb") as f:
    pickle.dump(X_train.columns, f  )

