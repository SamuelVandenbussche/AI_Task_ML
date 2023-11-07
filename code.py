import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Load the data
data = pd.read_csv('adult.data', skiprows=1, names=[
    'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
    'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
])

# Data preprocessing
data = data.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

less_than_50k = data[data['income'] == '<=50K']
more_than_50k = data[data['income'] == '>50K']
less_than_50k = less_than_50k.sample(n=len(more_than_50k), random_state=42)
data_2 = pd.concat([less_than_50k, more_than_50k])

data_before_preprocessing = data_2.copy()

data_2.replace('?', np.nan, inplace=True)
data_2 = data_2.fillna(data_2.mode().iloc[0])
data_2 = pd.get_dummies(data_2, columns=['workclass','education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country'])
data_after_preprocessing = data_2.copy()  # Copy the preprocessed data for visualization

less_than_50k = data[data['income'] == '<=50K']
more_than_50k = data[data['income'] == '>50K']
less_than_50k = less_than_50k.sample(n=len(more_than_50k), random_state=42)
data_2 = pd.concat([less_than_50k, more_than_50k])

# Split the data
X = data_2.drop('income', axis=1)
y = data_2['income']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create Streamlit app
st.title("Machine Learning Model Comparison App")

st.write("I have chosen the adult dataset, this dataset was made to try and predict if a person has a yearly income that is above or below 50k dollars. It has attributes such as age, sex, nation of birth, education level and more.")

# EDA section
st.header("Exploratory Data Analysis")

# Bar chart of income levels
st.subheader("Distribution of Income Levels before undersampeling")
income_counts = data['income'].value_counts()
st.bar_chart(income_counts)
st.write("Bar chart of income levels")

# Bar chart of income levels
st.subheader("Distribution of Income Levels before undersampeling")
income_counts = data_2['income'].value_counts()
st.bar_chart(income_counts)
st.write("Bar chart of income levels")

# Display the original data and preprocessed data
st.subheader("Original Data")
st.write(data_before_preprocessing)

# Display data preprocessing steps
st.subheader("Data Preprocessing")

st.write("1. Filling Null Values:")
st.write("Null values are filled with the mode (most frequent value) for each column.")

st.write("2. One-Hot Encoding:")
st.write("Categorical columns are one-hot encoded to convert them into numerical format.")

st.subheader("Preprocessed Data")
st.write(data_after_preprocessing)

# Model selection section
st.header("Model Selection")
selected_model = st.selectbox("Select a machine learning model", ["Logistic Regression", "Gradient Boosting", "SVM"])

# Train and evaluate the selected model
if selected_model == "Logistic Regression":
    model = LogisticRegression()
elif selected_model == "Gradient Boosting":
    model = GradientBoostingClassifier()
else:
    model = SVC()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"{selected_model} Model Accuracy: {accuracy:.2f}")

st.subheader("Confusion Matrix")
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
st.pyplot(plt)

