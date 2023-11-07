import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit as st

# Load the data
data = pd.read_csv('adult.data', skiprows=1, names=[
    'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
    'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
])

# Data preprocessing
data = data.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
data.replace('?', np.nan, inplace=True)
data = data.fillna(data.mode().iloc[0])
data = pd.get_dummies(data, columns=['workclass','education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country'])

# Split the data
X = data.drop('income', axis=1)
y = data['income']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create Streamlit app
st.title("Machine Learning Model Comparison App")

# EDA section
st.header("Exploratory Data Analysis")

# Numerical feature histogram
st.subheader("Numerical Feature Histograms")
numerical_features = data.select_dtypes(include=[np.number]).columns
selected_feature = st.selectbox("Select a numerical feature", numerical_features)
st.pyplot(st.pyplot(data[selected_feature].plot.hist(bins=20)))

# Categorical feature bar chart
st.subheader("Categorical Feature Bar Chart")
categorical_features = data.select_dtypes(exclude=[np.number]).columns
selected_cat_feature = st.selectbox("Select a categorical feature", categorical_features)
st.bar_chart(data[selected_cat_feature].value_counts())

# Model selection section
st.header("Model Selection")
selected_model = st.selectbox("Select a machine learning model", ["Logistic Regression", "Random Forest", "SVM"])

# Train and evaluate the selected model
if selected_model == "Logistic Regression":
    model = LogisticRegression()
elif selected_model == "Random Forest":
    model = RandomForestClassifier()
else:
    model = SVC()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"{selected_model} Model Accuracy: {accuracy:.2f}")

st.write("Add more controls and visualizations for hyperparameter tuning and comparison here.")
