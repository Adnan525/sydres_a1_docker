import numpy as np
import pandas as pd
import geopandas
from shapely.geometry import Point
from sklearn.preprocessing import LabelEncoder
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

import warnings
warnings.filterwarnings("ignore")

df_full = pd.read_csv("data/zomato_df_final_data.csv")
df = df_full.drop(columns = ["address", "link", "phone", "title", "color", "cuisine_color", "rating_text", "lat", "lng"], axis = 1)
df.groupon = df.groupon.astype(int)
df['subzone'] = df['subzone'].apply(lambda x: x.split(',')[-1].strip() if ',' in x else x)

label_encoder = LabelEncoder()
df['subzone_encoded'] = label_encoder.fit_transform(df['subzone'])

df.dropna(subset=['rating_number'], inplace=True)

df['cost'] = df.groupby('subzone')['cost'].transform(lambda x: x.fillna(x.mean()))
df['cost_2'] = df.groupby('subzone')['cost_2'].transform(lambda x: x.fillna(x.mean()))
df = df.drop(columns = ["subzone"], axis = 1)

df = df.drop(columns = ["cuisine"])
most_frequent = df["type"].mode()[0]
df["type"].fillna(most_frequent, inplace=True)

types = set()
#custom function to check all the values
def get_type(str):
    pattern = r',\s*' #removing comma(,) followed by immediate space
    cleaned_text = re.sub(pattern, '-', str)
    pattern2 = r'[\'\[\],]'
    cleaned_text = re.sub(pattern2, "", cleaned_text)
    temp = cleaned_text.split("-")
    for type in temp:
        types.add(type)
for value in df["type"]:
    get_type(value)
for dining_type in types:
    df[f"is_{dining_type}"] = df['type'].apply(lambda x: 1 if dining_type in x else 0)
df.drop('type', axis=1, inplace=True)

# model 1
X = df.drop(columns=['rating_number'])  # Features
y = df['rating_number']  # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model_regression_1 = LinearRegression()
model_regression_1.fit(X_train, y_train)

# predictions
y_pred = model_regression_1.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("=======================")
print("model_regression_1")
print("Mean Squared Error:", mse)
print("R-squared:", r2)
print("=======================")
print("")

# model 2
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_regression_2 = SGDRegressor(loss="squared_error", max_iter=1000, random_state=0)
model_regression_2.fit(X_train_scaled, y_train)

y_pred = model_regression_2.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("=======================")
print("model_regression_2")
print("Mean Squared Error:", mse)
print("R-squared:", r2)
print("=======================")
print("")

# model 3 classification
df_full.dropna(subset=["rating_text"], inplace=True)
df["rating_text"] = df_full["rating_text"]

class_mapping = {
    "Poor": 1,
    "Average": 1,
    "Good": 2,
    "Very Good": 2,
    "Excellent": 2
}
df["binary_rating"] = df["rating_text"].map(class_mapping)
df = df.drop(columns = ["rating_text"], axis = 1)

X = df.drop(columns=['binary_rating'])  # Features
y = df['binary_rating']  # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

classification_model_logistic_regression_3 = LogisticRegression()
classification_model_logistic_regression_3.fit(X_train, y_train)

# predictions
y_pred = classification_model_logistic_regression_3.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
print("=======================")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC Score:", roc_auc)
print("Confusion Matrix:\n", conf_matrix)
print("=======================")