import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_csv("provider_fraud_detection_data.csv")

# Preprocess the data
label_encoder = LabelEncoder()
data[''] = label_encoder.fit_transform(data[''])

# Split the data into train and test sets
X = data.drop('', axis=1)
y = data['']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Define the app title and header
st.title("Medical Fraud Detection")
st.header("Fraud Prediction")

# Add input fields for user input
patient_id = st.text_input("Patient ID")
claim_amount = st.number_input("Claim Amount")
provider_id = st.text_input("Provider ID")
age = st.number_input("Age")
is_inpatient = st.selectbox("Is Inpatient", ["Yes", "No"])
is_outpatient = st.selectbox("Is Outpatient", ["Yes", "No"])

# Create a dictionary from user input
input_data = {
    'patient_id': [patient_id],
    'claim_amount': [claim_amount],
    'provider_id': [provider_id],
    'age': [age],
    'is_inpatient': [is_inpatient],
    'is_outpatient': [is_outpatient]
}

# Create a DataFrame from the input data
input_df = pd.DataFrame(input_data)

# Make predictions on user input
prediction = model.predict(input_df)[0]

# Convert prediction label back to original class
prediction_label = label_encoder.inverse_transform([prediction])[0]

# Display the prediction result
st.subheader("Prediction Result")
st.write(f"The predicted fraud label for the input is: {prediction_label}")

# Evaluate the model
test_preds = model.predict(X_test)
accuracy = accuracy_score(y_test, test_preds)
precision = precision_score(y_test, test_preds)
recall = recall_score(y_test, test_preds)
f1 = f1_score(y_test, test_preds)

# Display the evaluation metrics
st.header("Model Evaluation")
st.subheader("Accuracy: {:.2%}".format(accuracy))
st.subheader("Precision: {:.2%}".format(precision))
st.subheader("Recall: {:.2%}".format(recall))
st.subheader("F1 Score: {:.2%}".format(f1))
