from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the trained model
with open('heart_disease_model.pkl', 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect form data from the request
        age = int(request.form['age'])
        sex = int(request.form['sex'])
        cp = int(request.form['cp'])  # Chest Pain Type
        trestbps = int(request.form['trestbps'])  # Resting Blood Pressure
        chol = int(request.form['chol'])  # Cholesterol
        fbs = int(request.form['fbs'])  # Fasting Blood Sugar
        restecg = int(request.form['restecg'])  # Resting Electrocardiographic Results
        thalach = int(request.form['thalach'])  # Maximum Heart Rate
        exang = int(request.form['exang'])  # Exercise Induced Angina
        oldpeak = float(request.form['oldpeak'])  # ST Depression
        slope = int(request.form['slope'])  # Slope of the Peak Exercise ST Segment
        ca = int(request.form['ca'])  # Number of Major Vessels
        thal = int(request.form['thal'])  # Thalassemia

        # Input validation (adjust ranges based on dataset)
        if age < 0 or sex not in [0, 1] or cp not in [0, 1, 2, 3] or \
           trestbps < 0 or chol < 0 or fbs not in [0, 1] or \
           restecg not in [0, 1, 2] or thalach < 0 or \
           exang not in [0, 1] or oldpeak < 0 or \
           slope not in [0, 1, 2] or ca < 0 or thal not in [1, 2, 3]:
            raise ValueError("Invalid input values. Please check your inputs.")

        # Create input array for prediction
        input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                                 thalach, exang, oldpeak, slope, ca, thal]])

        # Debugging output
        print("Input data for prediction:", input_data)

        # Predict using the loaded model
        prediction = model.predict(input_data)
        prediction_prob = model.predict_proba(input_data)  # Get the probabilities

        # Get the probability of having heart disease (assuming class 1 indicates heart disease)
        heart_disease_prob = prediction_prob[0][1]  # Probability of class 1
        print("Prediction:", prediction)
        print("Heart Disease Probability:", heart_disease_prob)

        # Output the result
        result = 'Heart Disease Detected' if prediction[0] == 1 else 'No Heart Disease'
        
        # Convert to percentage
        risk_percentage = heart_disease_prob * 100  

    except Exception as e:
        print("Error occurred:", e)
        result = f"An error occurred: {e}"
        risk_percentage = None

    return render_template('result.html', prediction_text=result, risk_percentage=risk_percentage)

if __name__ == "__main__":
    app.run(debug=True)
