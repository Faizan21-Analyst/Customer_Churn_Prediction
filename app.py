from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model (ensure that you have a trained model in .pkl format)
model = pickle.load(open('churn.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Collect data from form
        gender = 1 if request.form['Gender'] == 'Yes' else 0
        senior_citizen = 1 if request.form['SeniorCitizen'] == 'Yes' else 0
        partner = 1 if request.form['Partner'] == 'Yes' else 0
        dependents = 1 if request.form['Dependents'] == 'Yes' else 0
        tenure = int(request.form['tenure'])
        phone_service = 1 if request.form['PhoneService'] == 'Yes' else 0
        paperless_billing = 1 if request.form['PaperlessBilling'] == 'Yes' else 0
        monthly_charges = float(request.form['MonthlyCharges'])
        total_charges = float(request.form['TotalCharges'])

        # One-hot encode categorical features into binary variables
        multiple_lines_no_phone_service = 1 if request.form['MultipleLines'] == 'No phone service' else 0
        multiple_lines_yes = 1 if request.form['MultipleLines'] == 'Yes' else 0

        internet_service_fiber = 1 if request.form['InternetService'] == 'Fiber optic' else 0
        internet_service_no = 1 if request.form['InternetService'] == 'No' else 0

        online_security_no_internet = 1 if request.form['OnlineSecurity'] == 'No internet service' else 0
        online_security_yes = 1 if request.form['OnlineSecurity'] == 'Yes' else 0

        # Newly added features (OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies)
        online_backup_no_internet = 1 if request.form['OnlineBackup'] == 'No internet service' else 0
        online_backup_yes = 1 if request.form['OnlineBackup'] == 'Yes' else 0

        device_protection_no_internet = 1 if request.form['DeviceProtection'] == 'No internet service' else 0
        device_protection_yes = 1 if request.form['DeviceProtection'] == 'Yes' else 0

        tech_support_no_internet = 1 if request.form['TechSupport'] == 'No internet service' else 0
        tech_support_yes = 1 if request.form['TechSupport'] == 'Yes' else 0

        streaming_tv_no_internet = 1 if request.form['StreamingTV'] == 'No internet service' else 0
        streaming_tv_yes = 1 if request.form['StreamingTV'] == 'Yes' else 0

        streaming_movies_no_internet = 1 if request.form['StreamingMovies'] == 'No internet service' else 0
        streaming_movies_yes = 1 if request.form['StreamingMovies'] == 'Yes' else 0

        contract_one_year = 1 if request.form['Contract'] == 'One year' else 0
        contract_two_year = 1 if request.form['Contract'] == 'Two year' else 0

        payment_method_credit_card = 1 if request.form['PaymentMethod'] == 'Credit card (automatic)' else 0
        payment_method_electronic_check = 1 if request.form['PaymentMethod'] == 'Electronic check' else 0
        payment_method_mailed_check = 1 if request.form['PaymentMethod'] == 'Mailed check' else 0  # Newly added

        # Collect all binary-encoded data in the correct order
        data = np.array([[gender, senior_citizen, partner, dependents, tenure, phone_service,
                          paperless_billing, monthly_charges, total_charges,
                          multiple_lines_no_phone_service, multiple_lines_yes,
                          internet_service_fiber, internet_service_no,
                          online_security_no_internet, online_security_yes,
                          online_backup_no_internet, online_backup_yes,
                          device_protection_no_internet, device_protection_yes,
                          tech_support_no_internet, tech_support_yes,
                          streaming_tv_no_internet, streaming_tv_yes,
                          streaming_movies_no_internet, streaming_movies_yes,
                          contract_one_year, contract_two_year,
                          payment_method_credit_card, payment_method_electronic_check, 
                          payment_method_mailed_check]])  # Added mailed check here

        # Perform prediction
        prediction = model.predict(data)[0]

        # Interpret the result
        result = 'Churn' if prediction == 1 else 'No Churn'
        return render_template('result.html', prediction=result)


if __name__ == '__main__':
    app.run(debug=True)
