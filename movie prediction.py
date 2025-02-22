#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
file_path = "G:\Athrav Project\MoviePrediction\IMDb Movies India.csv"  # Update with your dataset path
df = pd.read_csv(file_path, encoding="ISO-8859-1")

# Select relevant columns
df = df[['Name', 'Year', 'Duration', 'Genre', 'Rating', 'Votes', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']]

# Drop rows where Rating (target variable) is missing
df.dropna(subset=['Rating'], inplace=True)

# Extract numerical year from string
df['Year'] = df['Year'].str.extract('(\d{4})').astype(float)

# Convert 'Votes' to numeric by removing commas
df['Votes'] = df['Votes'].str.replace(',', '', regex=True).astype(float)

# Extract only the numeric value from 'Duration' column
df['Duration'] = df['Duration'].str.extract('(\d+)').astype(float)

# Fill missing values with median values (to avoid dropping too much data)
df['Year'].fillna(df['Year'].median(), inplace=True)
df['Duration'].fillna(df['Duration'].median(), inplace=True)
df['Votes'].fillna(0, inplace=True)  # Votes can be 0 if no one voted

# Encoding categorical features
label_encoders = {}
for col in ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']:
    label_encoders[col] = LabelEncoder()
    df[col] = label_encoders[col].fit_transform(df[col].astype(str))

# Define features and target
X = df[['Year', 'Duration', 'Votes', 'Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']]
y = df['Rating']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate model
# mse = mean_squared_
mse = mean_squared_error(y_test, y_pred)


# In[10]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
file_path = "G:\Athrav Project\MoviePrediction\IMDb Movies India.csv"  # Update with your dataset path
df = pd.read_csv(file_path, encoding="ISO-8859-1")

# Select relevant columns
df = df[['Name', 'Year', 'Duration', 'Genre', 'Rating', 'Votes', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']]

# Drop rows where Rating (target variable) is missing
df.dropna(subset=['Rating'], inplace=True)

# Extract numerical year from string
df['Year'] = df['Year'].str.extract('(\d{4})').astype(float)

# Convert 'Votes' to numeric by removing commas
df['Votes'] = df['Votes'].str.replace(',', '', regex=True).astype(float)

# Extract only the numeric value from 'Duration' column
df['Duration'] = df['Duration'].str.extract('(\d+)').astype(float)

# Fill missing values with median values (to avoid dropping too much data)
df['Year'].fillna(df['Year'].median(), inplace=True)
df['Duration'].fillna(df['Duration'].median(), inplace=True)
df['Votes'].fillna(0, inplace=True)  # Votes can be 0 if no one voted

# Encoding categorical features
label_encoders = {}
for col in ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']:
    label_encoders[col] = LabelEncoder()
    df[col] = label_encoders[col].fit_transform(df[col].astype(str))

# Define features and target
X = df[['Year', 'Duration', 'Votes', 'Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']]
y = df['Rating']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate model
# mse = mean_squared_
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Feature Importance Visualization
plt.figure(figsize=(8,5))
sns.barplot(x=model.feature_importances_, y=X.columns)
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Feature Importance in Movie Rating Prediction")
plt.show()


# In[ ]:





# In[ ]:




