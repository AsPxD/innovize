import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
df = pd.read_csv('innovize_final_ml.csv')

# Data Cleaning and Preprocessing
# Handle missing values
for col in df.columns:
    if df[col].isnull().any():
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])  # Mode for categorical
        else:
            df[col] = df[col].fillna(df[col].median())  # Median for numerical

# Convert categorical features to numerical using Label Encoding
categorical_cols = ['diet_pref', 'act_level', 'career', 'gender']
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# Define features (X) and target (y)
X = df[['phy_fitness', 'mindfulness', 'diet_pref', 'act_level', 'sleep_hrs', 'career', 'gender', 'daily_avg_steps', 'daily_avg_calories']]
y = df['is_healthy']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Hyperparameter Tuning for Random Forest using GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300], # Number of trees
    'max_depth': [4, 5, 6, 7, 8],     # Maximum depth of trees
    'min_samples_split': [2, 4],  # Minimum samples to split
    'min_samples_leaf': [1, 2]      # Minimum samples in leaf
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best model from GridSearchCV
best_model = grid_search.best_estimator_

# Make predictions on the test set using the best model
y_pred = best_model.predict(X_test)

# Evaluate the best model
accuracy = accuracy_score(y_test, y_pred)
print(f"Best Model (Random Forest with GridSearchCV) Accuracy: {accuracy:.4f}")
print(f"Best Parameters: {grid_search.best_params_}")

# Visualization (using the best model)
X_vis = df[['phy_fitness', 'mindfulness']]
y_vis = df['is_healthy']

# Split the visualization data
X_train_vis, X_test_vis, y_train_vis, y_test_vis = train_test_split(X_vis, y_vis, test_size=0.2, random_state=42)

# Scale the visualization data
X_train_vis = scaler.fit_transform(X_train_vis)
X_test_vis = scaler.transform(X_test_vis)


#Train a visualization Model for visualization to reduce dimensionality.
best_model_vis = RandomForestClassifier(random_state=42, **grid_search.best_params_)
best_model_vis.fit(X_train_vis, y_train_vis)

# Create a grid of points to plot the decision boundary
x_min, x_max = X_vis['phy_fitness'].min() - 1, X_vis['phy_fitness'].max() + 1
y_min, y_max = X_vis['mindfulness'].min() - 1, X_vis['mindfulness'].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# Predict the class for each point in the grid
Z = best_model_vis.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.contourf(xx, yy, Z, alpha=0.4)

# Plot the data points
plt.scatter(X_vis['phy_fitness'], X_vis['mindfulness'], c=y_vis, s=20, edgecolor='k')
plt.xlabel('Physical Fitness')
plt.ylabel('Mindfulness')
plt.title(f'Decision Boundary of Tuned Random Forest')
plt.show()