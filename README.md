# DataVisualizationProject
🧠 This project demonstrates the construction of a machine learning pipeline using Scikit-learn to predict apartment rental prices in the United States. The workflow follows a simplified version of the end-to-end machine learning process, including data preprocessing, pipeline creation, model training, and evaluation.

📌 Project Overview

The objective of this assignment is to build a multiple linear regression model (and compare with a Random Forest model) to predict apartment prices based on various features such as location, size, and amenities.

The project emphasizes:

Data preprocessing using pipelines
Feature engineering
Model training and evaluation
Proper machine learning workflow structure

📂 Dataset
The dataset consists of 10,000 apartment listings with 10 features, including:

id – Unique identifier (removed during preprocessing)
latitude – Geographic location
longitude – Geographic location
bathrooms – Number of bathrooms
bedrooms – Number of bedrooms
fee – Fee indicator (removed during preprocessing)
has_photo – Listing includes photo (Yes/No)
pets_allowed – Allowed pets (cats, dogs, etc.)
square_feet – Apartment size
price – Target variable

⚙️ Technologies Used
Python
Pandas
NumPy
Scikit-learn
Matplotlib

🔄 Machine Learning Workflow
1. Data Preparation
Removed irrelevant columns (id, fee)
Split data into training and testing sets
Handled missing values
2. Pipeline Construction
Numeric Pipeline
Median imputation
Standard scaling
Categorical Pipeline
Constant imputation ("No_Pets")
One-hot encoding (drop first category)
Column Transformer
Combined numeric and categorical pipelines for preprocessing

🤖 Models Used
1. Linear Regression
Baseline model
Evaluated using 10-fold cross-validation
2. Random Forest Regressor
Improved performance over linear regression
Reduced RMSE significantly

📊 Evaluation Metric
Root Mean Squared Error (RMSE)
Cross-validation used for model comparison
Final evaluation performed on test dataset
