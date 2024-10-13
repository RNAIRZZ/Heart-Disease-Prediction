from pyexpat import model
from urllib import request


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect form data from the request
        age = int(request.form['age'])
        sex = int(request.form['sex'])
        cp = int(request.form['cp'])
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = int(request.form['fbs'])
        restecg = int(request.form['restecg'])
        thalach = int(request.form['thalach'])
        exang = int(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        slope = int(request.form['slope'])
        ca = int(request.form['ca'])
        thal = int(request.form['thal'])

        # Create input array for prediction
        input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

        # Debugging output
        print("Input data:", input_data)

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
        result = f"An error occurred: {e}"
        risk_percentage = None

    return render_template('result.html', prediction_text=result, risk_percentage=risk_percentage)
