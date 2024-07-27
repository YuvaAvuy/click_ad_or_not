import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load the cleaned dataset
data = pd.read_csv("cleaned_ad_data.csv")

# Ensure 'Clicked on Ad' is in binary format
data["Clicked on Ad"] = data["Clicked on Ad"].map({"No": 0, "Yes": 1})

# Prepare data for machine learning model
features = ["Daily Time Spent on Site", "Age", "Area Income", "Daily Internet Usage", "Gender"]
x = data[features]
y = data["Clicked on Ad"]

# Train the Random Forest model
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=4)
model = RandomForestClassifier(random_state=4)
model.fit(xtrain, ytrain)

# Define the Streamlit app
def main():
    st.title('Ads Click Through Rate Prediction')

    # Collect user input
    a = st.slider("Daily Time Spent on Site", float(x["Daily Time Spent on Site"].min()), float(x["Daily Time Spent on Site"].max()))
    b = st.slider("Age", float(x["Age"].min()), float(x["Age"].max()))
    c = st.slider("Area Income", float(x["Area Income"].min()), float(x["Area Income"].max()))
    d = st.slider("Daily Internet Usage", float(x["Daily Internet Usage"].min()), float(x["Daily Internet Usage"].max()))
    e = st.selectbox("Gender", ["Male", "Female"])

    gender_mapping = {"Male": 1, "Female": 0}
    e = gender_mapping[e]

    # Make prediction
    features = np.array([[a, b, c, d, e]])
    prediction = model.predict(features)[0]

    st.subheader("Prediction Result:")
    if prediction == 1:
        st.write("User is likely to click on the ad.")
    else:
        st.write("User is not likely to click on the ad.")

if __name__ == '__main__':
    main()
