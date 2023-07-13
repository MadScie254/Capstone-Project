import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

def main():
    # st.title('Healthcare Provider Fraud Detector')
    html_temp="""
    <div style="background-color:teal;padding:10px">
    <h2 style="color:white;text-align:center;">Healthcare Provider Fraud Detector App </h2>
   """ 
    # Render the HTML content
    st.markdown(html_temp, unsafe_allow_html=True)
    
    # Upload CSV file
    uploaded_file = st.file_uploader("Upload CSV file", type="csv")
    
    if uploaded_file is not None:
        # Read the uploaded file as a DataFrame
        df = pd.read_csv(uploaded_file)
        
        # Perform preprocessing on the data
        processed_data = preprocess_data(df)
        
        # Load the trained model
        with open('xgboost_model_reduced_features.pkl', 'rb') as file:
            model = pickle.load(file)
        
        # Make predictions
        predictions = model.predict(processed_data)
        
        # Count fraudulent activities
        fraud_count = sum(predictions)
        
        # Calculate percentages
        total_samples = len(predictions)
        fraud_percentage = (fraud_count / total_samples) * 100
        
        # Display the predicted output
        st.write("Predicted Output:")
        st.write(predictions)
        
        # Display the fraudulent activity count and percentage
        st.write("Fraudulent Activities:")
        st.write("Count:", fraud_count)
        st.write("Percentage:", fraud_percentage, "%")
        
        # Get the rows with fraudulent activities
        fraud_rows = df[predictions == 1]
        
        # Display the fraudulent rows
        if not fraud_rows.empty:
            st.write("Fraudulent Rows:")
            st.write(fraud_rows)
        else:
            st.write("No fraudulent activities detected.")

        # Plot a bar chart of fraudulent activities
        plot_fraud_chart(predictions)

        # Check if any fraudulent activities are detected
        if fraud_count > 0:
            # Display the "Predict" button
            if st.button("Predict"):
                # Perform any additional actions you want to take on button click
                pass

def preprocess_data(df):
    # Perform preprocessing steps on the DataFrame
    # Ensure that the preprocessing steps match those used during training
    
    # Identify categorical columns
    categorical_cols = df.select_dtypes(include="object").columns
    
    # Encode categorical columns
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    
    # Apply feature scaling if required
    
    return df

def plot_fraud_chart(predictions):
    # Create a DataFrame for plotting
    df = pd.DataFrame(predictions, columns=["Fraudulent"])
    
    # Plot a bar chart
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x="Fraudulent")
    plt.title("Fraudulent Activities")
    plt.xlabel("Fraudulent")
    plt.ylabel("Count")
    st.pyplot()

if __name__ == "__main__":
    main()