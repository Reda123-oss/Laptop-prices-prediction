1. Introduction
The project addresses a common problem for consumers: the overwhelming number of choices in the ever-changing laptop market. The goal is to develop an intelligent application that simplifies this decision-making process by predicting laptop prices and recommending models based on their technical specifications.
2. Objectives
The application is designed to have two main functions:
Mode 1 (Price Prediction): A user enters the desired technical specifications (CPU, RAM, storage, etc.), and the model predicts the estimated price.
Mode 2 (Laptop Recommendation): A user enters their budget, and the application suggests a list of laptops that fit that price range.
The project also aims to build an intuitive web interface using the Flask framework and to base the predictions on a real-world dataset.
3. Dataset
The model is built on a detailed dataset of laptops. Key features include:
Brand, RAM, Storage (Type and Capacity)
CPU, GPU (Company and Model)
Screen size, OS, and special options (Touchscreen, Retina Display)
Price (the target variable for prediction)
The team started with 23 columns of data and, after analysis, reduced it to the 17 most impactful features to improve model performance.
4. Realization (Implementation & Results)
This section details the machine learning process:
Data Preparation: They cleaned the data, converted categorical features (like brand names) into numerical values, and split the dataset into 80% for training the model and 20% for testing it.
Model Training: They experimented with several supervised regression models, including Linear Regression, Random Forest, SVR, and XGBoost.
Performance: The XGBoost Regressor model proved to be the most effective, achieving an R² score of approximately 0.87. This means the model can explain 87% of the variance in laptop prices, indicating a strong predictive capability.
Model Selection: After using GridSearch to optimize hyperparameters, they confirmed XGBoost as the best model and saved it for use in the application.
5. Technologies and Chosen Model
The project utilized a modern tech stack:
Backend: Flask (Python web framework)
Machine Learning: Scikit-learn, Pandas, XGBoost
Frontend: HTML, JavaScript, Bootstrap
Model Persistence: Joblib
The final chosen model for the application is XGBoost.
