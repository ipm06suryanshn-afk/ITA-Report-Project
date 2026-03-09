
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import r2_score, accuracy_score, classification_report
from sklearn.cluster import KMeans

# 1. Data Loading & Preprocessing
df = pd.read_csv('/content/startup_funding.csv')
df = df.drop(columns=['Unnamed: 9'])
df['Amount in USD'] = df['Amount in USD'].str.replace(',', '', regex=False)
df['Amount in USD'] = pd.to_numeric(df['Amount in USD'], errors='coerce')
df['Amount in USD'] = df['Amount in USD'].fillna(df['Amount in USD'].median())
df['City  Location'] = df['City  Location'].fillna('Unknown')
df['Industry Vertical'] = df['Industry Vertical'].fillna('Unknown')
df['InvestmentnType'] = df['InvestmentnType'].fillna('Unknown')

le_industry, le_location, le_type = LabelEncoder(), LabelEncoder(), LabelEncoder()
df['Industry_Encoded'] = le_industry.fit_transform(df['Industry Vertical'].astype(str))
df['Location_Encoded'] = le_location.fit_transform(df['City  Location'].astype(str))
df['Stage_Encoded'] = le_type.fit_transform(df['InvestmentnType'].astype(str))

# 2. Supervised Learning: Regression
X = df[['Industry_Encoded', 'Location_Encoded', 'Stage_Encoded']]
y = df['Amount in USD']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr, rf, gb = LinearRegression(), RandomForestRegressor(random_state=42), GradientBoostingRegressor(random_state=42)
lr.fit(X_train, y_train)
rf.fit(X_train, y_train)
gb.fit(X_train, y_train)

# 3. Supervised Learning: Classification
target_stages = ['Seed Round', 'Series A', 'Series B']
df_f = df[df['InvestmentnType'].isin(target_stages)].copy()
X_c = df_f[['Industry_Encoded', 'Location_Encoded', 'Amount in USD']]
y_c = df_f['InvestmentnType']
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_c, y_c, test_size=0.2, random_state=42)

rf_c = RandomForestClassifier(random_state=42).fit(X_train_c, y_train_c)

# 4. Unsupervised Learning: Clustering
X_cluster = df[['Amount in USD', 'Industry_Encoded', 'Location_Encoded']]
X_scaled = StandardScaler().fit_transform(X_cluster)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster_ID'] = kmeans.fit_predict(X_scaled)

print("Processing Complete. Models Trained.")
