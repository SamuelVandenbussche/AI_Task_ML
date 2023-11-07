import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data = pd.read_csv('adult.data', skiprows=1, names=[
    'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
    'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
])

# Data preprocessing
data = data.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
data.replace('?', np.nan, inplace=True)
data_before_preprocessing = data.copy()  # Copy the original data for visualization
data = data.fillna(data.mode().iloc[0])
data = pd.get_dummies(data, columns=['workclass','education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country'])
data_after_preprocessing = data.copy()  # Copy the preprocessed data for visualization

# Split the data
X = data.drop('income', axis=1)
y = data['income']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create Streamlit app
st.title("Machine Learning Model Comparison App")

# EDA section
st.header("Exploratory Data Analysis")

# Display data preprocessing steps
st.subheader("Data Preprocessing")

st.write("1. Filling Null Values:")
st.write("Null values are filled with the mode (most frequent value) for each column.")

st.write("2. One-Hot Encoding:")
st.write("Categorical columns are one-hot encoded to convert them into numerical format.")

# Display the original data and preprocessed data
st.subheader("Original Data")
st.write(data_before_preprocessing)

st.subheader("Preprocessed Data")
st.write(data_after_preprocessing)

# Numerical feature histogram
st.subheader("Numerical Feature Histograms")
numerical_features = data.select_dtypes(include=[np.number]).columns
selected_feature = st.selectbox("Select a numerical feature", numerical_features, key="histogram_select_feature")

# Create a Matplotlib figure and axis
fig, ax = plt.subplots()
ax.hist(data[selected_feature], bins=20)
st.pyplot(fig)  # Display the Matplotlib figure

st.write("Histogram of", selected_feature)

# Box plots
st.subheader("Box Plots of Numerical Features")
selected_box_feature = st.selectbox("Select a numerical feature", numerical_features, key="box_select_feature")

# Create a seaborn box plot
fig, ax = plt.subplots()
sns.boxplot(x=data[selected_box_feature], ax=ax)
st.pyplot(fig)
st.write("Box plot of", selected_box_feature)


# Categorical feature bar chart
st.subheader("Categorical Feature Bar Charts")
categorical_features = data.select_dtypes(exclude=[np.number]).columns
selected_cat_feature = st.selectbox("Select a categorical feature", categorical_features)
st.bar_chart(data[selected_cat_feature].value_counts())
st.write("Bar chart of", selected_cat_feature)

# Scatter plots
st.subheader("Scatter Plots")
scatter_x = st.selectbox("Select a numerical feature for the x-axis", numerical_features, key="scatter_x")
scatter_y = st.selectbox("Select a numerical feature for the y-axis", numerical_features, key="scatter_y")
st.pyplot(plt.scatter(data[scatter_x], data[scatter_y]))
st.write("Scatter plot of", scatter_x, "vs", scatter_y)

# Correlation heatmap
st.subheader("Correlation Heatmap")
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True)
st.pyplot(plt)
st.write("Correlation heatmap")

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
