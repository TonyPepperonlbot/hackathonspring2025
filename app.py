from flask import Flask, render_template, request, redirect, url_for
import json
import os
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

DATA_FILE = "survey_data.json"


if os.path.exists(DATA_FILE):
    with open(DATA_FILE, "r") as f:
        try:
            survey_data = json.load(f)
        except json.JSONDecodeError:
            survey_data = []
else:
    survey_data = []


with open("logistic_regression_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')

@app.route('/survey', methods=['GET', 'POST'])
def survey():
    if request.method == 'POST':
        # Get form data
        new_entry = {
            "Fever": request.form.get("Fever", ""),
            "Cough": request.form.get("Cough", ""),
            "Fatigue": request.form.get("Fatigue", ""),
            "Difficulty_Breathing": request.form.get("Difficulty_Breathing", ""),
            "Gender": request.form.get("Gender", ""),
            "Blood_Pressure": request.form.get("Blood_Pressure", ""),
            "Cholesterol_Level": request.form.get("Cholesterol_Level", "")
        }


        survey_data.append(new_entry)


        with open(DATA_FILE, "w") as f:
            json.dump(survey_data, f, indent=4)

        return redirect(url_for('survey_confirmation', entry_index=len(survey_data) - 1))

    return render_template('survey.html')

@app.route('/survey_confirmation/<int:entry_index>', methods=['GET'])
def survey_confirmation(entry_index):

    entry = survey_data[entry_index]


    input_df = pd.DataFrame([entry])


    input_df = input_df.rename(columns={
        "Blood Pressure": "Blood_Pressure",
        "Cholesterol Level": "Cholesterol_Level"
    })


    for col in ["Fever", "Cough", "Fatigue", "Difficulty_Breathing"]:
        if col in input_df.columns:
            input_df[col] = input_df[col].map({"No": 0, "Yes": 1})

    categorical_columns = ["Gender", "Blood_Pressure", "Cholesterol_Level"]
    input_df = pd.get_dummies(input_df, columns=categorical_columns)


    from data import X_resampled  
    missing_cols = set(X_resampled.columns) - set(input_df.columns)
    for col in missing_cols:
        input_df[col] = 0  

    input_df = input_df[X_resampled.columns] 


    probabilities = model.predict_proba(input_df)[0]


    top_3 = probabilities.argsort()[-3:][::-1]
    top_disease = label_encoder.inverse_transform(top_3)
    top_probabilities = probabilities[top_3]


    top_diseases_with_probs = list(zip(top_disease, top_probabilities))


    return render_template('survey_confirmation.html', entry=entry, top_diseases_with_probs=top_diseases_with_probs)

if __name__ == '__main__':
    app.run(debug=False, port=5001)  
