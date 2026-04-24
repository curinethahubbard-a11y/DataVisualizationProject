import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#Load dataset
df = pd.read_csv('day.csv')
df.head()

#Drop unnecessary columns
df = df.drop(['instant', 'dteday', 'casual', 'registered'], axis=1)

#Map season to labels
season_map = {1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'}
df['season'] = df['season'].map(season_map)

#Check missing values
df.isnull().sum()

#Visualize correlations
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

#Boxplot of rentals by season
sns.boxplot(x='season', y='cnt', data=df)
plt.title("Bike Rentals by Season")
plt.show()

#Encode categorical variables
categorical_cols = ['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

#Normalize numerical columns
scaler = StandardScaler()
num_cols = ['temp', 'atemp', 'hum', 'windspeed']
df[num_cols] = scaler.fit_transform(df[num_cols])

#Split features and target
X = df.drop('cnt', axis=1)
y = df['cnt']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#Linear Regression
linreg = LinearRegression()
linreg.fit(X_train, y_train)
y_pred = linreg.predict(X_test)

#Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

#Plot predictions
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel("Actual Count")
plt.ylabel("Predicted Count")
plt.title("Actual vs Predicted Bike Rental Count")
plt.plot([0, max(y_test)], [0, max(y_test)], color='red', linestyle='--')
plt.show()

#K-Means Clustering
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

cluster_data = df[['temp', 'hum', 'windspeed', 'cnt']]
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(cluster_data)

#Silhouette Score
sil_score = silhouette_score(cluster_data, df['cluster'])
print(f"Silhouette Score: {sil_score:.2f}")

#Visualize Clusters
pca = PCA(n_components=2)
pca_components = pca.fit_transform(cluster_data)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=pca_components[:, 0], y=pca_components[:, 1], hue=df['cluster'], palette='Set2')
plt.title("K-Means Clusters Visualized with PCA")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend(title='Cluster')
plt.show()



#Save model
import joblib
joblib.dump(linreg, 'bike_demand_model.pkl')

#Predict demand function
def predict_demand(input_features, model):
    return model.predict(input_features)
    