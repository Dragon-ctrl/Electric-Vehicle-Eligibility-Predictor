import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import joblib

# Load preprocessed data
df = pd.read_csv('preprocessed_data.csv')

# Split data into training and testing sets
X = df.drop(['Clean Alternative Fuel Vehicle (CAFV) Eligibility'], axis=1)
y = df['Clean Alternative Fuel Vehicle (CAFV) Eligibility']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the models to test
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree Regressor': DecisionTreeRegressor(random_state=42),
    'Random Forest Regressor': RandomForestRegressor(random_state=42),
    'Support Vector Regressor': SVR(gamma='scale')
}

# Define the hyperparameters to tune for each model
parameters = {
    'Linear Regression': {},
    'Decision Tree Regressor': {'max_depth': [5, 10, 20, None], 'min_samples_split': [2, 5, 10]},
    'Random Forest Regressor': {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 20, None], 'min_samples_split': [2, 5, 10]},
    'Support Vector Regressor': {'kernel': ['linear', 'rbf'], 'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}
}

best_model = None
best_score = -float('inf')

# Loop through each model and perform hyperparameter tuning using grid search
for name, model in models.items():
    grid_search = GridSearchCV(model, parameters[name], scoring='neg_mean_squared_error', cv=5, n_jobs=-1, refit=True)
    grid_search.fit(X_train, y_train)
    score = mean_squared_error(y_test, grid_search.predict(X_test))
    print(f'{name} - Best Parameters: {grid_search.best_params_}, Best Score: {score}')
    if best_score is None or score < best_score:
        best_score = score
        best_model = grid_search.best_estimator_

if best_model is not None:
    # Save the trained model as a joblib file
    joblib.dump(best_model, 'model.joblib')
else:
    print("No model was trained.")