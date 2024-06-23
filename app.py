from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

app = Flask(__name__)

# Load the model and scaler
with open('model/your_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('model/scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Define a route to render the HTML form
@app.route('/')
def index():
    return render_template('index.html')

# Define a route to handle prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve the input data from the HTML form
    age = float(request.form['age'])
    wtkg = float(request.form['wtkg'])
    hemo = float(request.form['hemo'])
    homo = float(request.form['homo'])
    drugs = float(request.form['drugs'])
    oprior = float(request.form['oprior'])
    race = float(request.form['race'])
    gender = float(request.form['gender'])
    strat = float(request.form['strat'])
    symptom = float(request.form['symptom'])
    treat = float(request.form['treat'])

    # Create a DataFrame from the input data
    input_data = pd.DataFrame({
        'age': [age],
        'wtkg': [wtkg],
        'hemo': [hemo],
        'homo': [homo],
        'drugs': [drugs],
        'oprior': [oprior],
        'race': [race],
        'gender': [gender],
        'strat': [strat],
        'symptom': [symptom],
        'treat': [treat]
    })

    # Scale the input data using the pre-fitted scaler
    scaled_input = scaler.transform(input_data)

    # Make predictions using the pre-fitted model
    prediction = model.predict(scaled_input)

    # Prepare response as JSON
    response = {'prediction': int(prediction[0])}  # Assuming prediction is binary (0 or 1)

    return render_template('result.html', prediction=int(prediction[0]))

if __name__ == '__main__':
    app.run(debug=True)
