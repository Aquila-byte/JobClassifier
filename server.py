from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
import csv

app = Flask(__name__)

model = joblib.load('model.pkl')

@app.route("/")
def hello_world():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the form
    experience = request.form['Experience']
    education = request.form['Education']
    employment = request.form['Employment']
    telecommuting = request.form['Telecommuting']
    logo = request.form['Logo']
    question = request.form['Question']
    
    
    my_dict ={"telecommuting": telecommuting, "has_company_logo": logo, "has_questions": question,
              "employment_type": employment, "required_experience": experience, "required_education": education}
    
    input_data = pd.DataFrame([my_dict])
    
    prediction = model.predict(input_data)
    my_list = prediction.tolist()
    prediction_proba = model.predict_proba(input_data)
    my_list2 = prediction_proba.tolist()
    
    with open('new_train_data.csv', 'a') as my_file:
        my_file.writelines(f"{telecommuting},{logo},{question},{employment},{experience},{education}\n")
        
    
    if my_list[0] == 1:
        answer = f"This is likely to be a Fraudulent Job Post with an accuracy of {(round(my_list2[0][1], 2) * 100)}%"
        return render_template('results.html', answer=answer)
    else:
        answer = f"This is a Valid Job Post with an accuracy of {(round(my_list2[0][0], 2) * 100)}% Apply Now"
        return render_template('results.html', answer=answer)
    
    
@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    feedback = request.form.get('feedback')
    
    with open('new_target_column.csv', 'a') as my_file:
        my_feedback = my_file.writelines(f"{feedback}\n")
    
    return render_template('results2.html')


