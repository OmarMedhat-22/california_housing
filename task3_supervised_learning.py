#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Task 3: Supervised Learning (Regression or Classification)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost as XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

# Set the style for the plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

def load_cleaned_data():
    """
    Load the cleaned data from Task 1
    
    Returns:
        pandas.DataFrame: The cleaned dataset
    """
    print("Loading cleaned data...")
    try:
        df_cleaned = pd.read_csv('california_housing_cleaned.csv')
        print(f"Cleaned data loaded with shape: {df_cleaned.shape}")
        return df_cleaned
    except FileNotFoundError:
        print("Cleaned data file not found. Please run Task 1 first.")
        return None

def prepare_data_for_supervised_learning(df):
    """
    Prepare the data for supervised learning
    
    Args:
        df (pandas.DataFrame): The cleaned dataset
        
    Returns:
        tuple: X (features), y (target), feature_names
    """
    print("\n--- Preparing Data for Supervised Learning ---")
    
    # Check for missing values
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print("\nDetected missing values in the dataset:")
        print(missing_values[missing_values > 0])
        
        # Handle missing values
        print("\nHandling missing values...")
        # For numerical columns, use median imputation
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        for col in numerical_cols:
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                print(f"  - Imputed {col} with median value: {median_val:.4f}")
        
        # For categorical columns, use most frequent value
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                mode_val = df[col].mode()[0]
                df[col] = df[col].fillna(mode_val)
                print(f"  - Imputed {col} with mode value: {mode_val}")
    
    # Define the target variable (median house value)
    y = df['MedHouseVal'].values
    
    # Define the features (all columns except the target)
    X = df.drop('MedHouseVal', axis=1)
    
    # Handle categorical variables
    X = pd.get_dummies(X, drop_first=True)
    
    # Store feature names for later use
    feature_names = X.columns.tolist()
    
    # Final check for any remaining NaN values
    if X.isnull().sum().sum() > 0:
        print("\nWarning: There are still NaN values in the features. Applying additional imputation.")
        # Simple imputation for any remaining NaNs
        X = X.fillna(X.mean())
    
    print(f"\nFeatures shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    return X, y, feature_names

def split_data(X, y):
    """
    Split the data into train, validation, and test sets
    
    Args:
        X (pandas.DataFrame): The features
        y (numpy.ndarray): The target
        
    Returns:
        tuple: X_train, X_val, X_test, y_train, y_val, y_test
    """
    print("\n--- Splitting Data ---")
    
    # First split: 80% train+val, 20% test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Second split: 75% train, 25% val (60% and 20% of the original data)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_baseline_model(X_train, y_train, X_val, y_val):
    """
    Train a baseline linear regression model
    
    Args:
        X_train (pandas.DataFrame): The training features
        y_train (numpy.ndarray): The training target
        X_val (pandas.DataFrame): The validation features
        y_val (numpy.ndarray): The validation target
        
    Returns:
        sklearn.linear_model.LinearRegression: The trained model
    """
    print("\n--- Training Baseline Linear Regression Model ---")
    
    # Create a pipeline with scaling and linear regression
    baseline_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ])
    
    # Train the model
    baseline_pipeline.fit(X_train, y_train)
    
    # Evaluate on training set
    y_train_pred = baseline_pipeline.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    
    print(f"Training RMSE: {train_rmse:.4f}")
    print(f"Training R²: {train_r2:.4f}")
    
    # Evaluate on validation set
    y_val_pred = baseline_pipeline.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_r2 = r2_score(y_val, y_val_pred)
    
    print(f"Validation RMSE: {val_rmse:.4f}")
    print(f"Validation R²: {val_r2:.4f}")
    
    # Plot actual vs. predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_val, y_val_pred, alpha=0.5)
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Baseline Model: Actual vs. Predicted Values')
    plt.savefig('baseline_actual_vs_predicted.png')
    
    return baseline_pipeline, val_rmse, val_r2

def train_ridge_regression(X_train, y_train, X_val, y_val):
    """
    Train a Ridge regression model with hyperparameter tuning
    
    Args:
        X_train (pandas.DataFrame): The training features
        y_train (numpy.ndarray): The training target
        X_val (pandas.DataFrame): The validation features
        y_val (numpy.ndarray): The validation target
        
    Returns:
        sklearn.pipeline.Pipeline: The trained model
    """
    print("\n--- Training Ridge Regression Model ---")
    
    # Create a pipeline with scaling and Ridge regression
    ridge_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', Ridge())
    ])
    
    # Define hyperparameter grid
    param_grid = {
        'model__alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
    }
    
    # Perform grid search with cross-validation
    grid_search = GridSearchCV(
        ridge_pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    # Get the best model
    best_ridge = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    print(f"Best hyperparameters: {best_params}")
    
    # Evaluate on training set
    y_train_pred = best_ridge.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    
    print(f"Training RMSE: {train_rmse:.4f}")
    print(f"Training R²: {train_r2:.4f}")
    
    # Evaluate on validation set
    y_val_pred = best_ridge.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_r2 = r2_score(y_val, y_val_pred)
    
    print(f"Validation RMSE: {val_rmse:.4f}")
    print(f"Validation R²: {val_r2:.4f}")
    
    return best_ridge, val_rmse, val_r2

def train_random_forest(X_train, y_train, X_val, y_val):
    """
    Train a Random Forest regression model with hyperparameter tuning
    
    Args:
        X_train (pandas.DataFrame): The training features
        y_train (numpy.ndarray): The training target
        X_val (pandas.DataFrame): The validation features
        y_val (numpy.ndarray): The validation target
        
    Returns:
        sklearn.pipeline.Pipeline: The trained model
    """
    print("\n--- Training Random Forest Regression Model ---")
    
    # Create a pipeline with Random Forest regression
    rf_pipeline = Pipeline([
        ('model', RandomForestRegressor(random_state=42))
    ])
    
    # Define hyperparameter grid
    param_grid = {
        'model__n_estimators': [50, 100, 200],
        'model__max_depth': [None, 10, 20, 30],
        'model__min_samples_split': [2, 5, 10]
    }
    
    # Perform grid search with cross-validation
    grid_search = GridSearchCV(
        rf_pipeline, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    # Get the best model
    best_rf = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    print(f"Best hyperparameters: {best_params}")
    
    # Evaluate on training set
    y_train_pred = best_rf.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    
    print(f"Training RMSE: {train_rmse:.4f}")
    print(f"Training R²: {train_r2:.4f}")
    
    # Evaluate on validation set
    y_val_pred = best_rf.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_r2 = r2_score(y_val, y_val_pred)
    
    print(f"Validation RMSE: {val_rmse:.4f}")
    print(f"Validation R²: {val_r2:.4f}")
    
    return best_rf, val_rmse, val_r2

def train_xgboost(X_train, y_train, X_val, y_val):
    """
    Train an XGBoost regression model with hyperparameter tuning
    
    Args:
        X_train (pandas.DataFrame): The training features
        y_train (numpy.ndarray): The training target
        X_val (pandas.DataFrame): The validation features
        y_val (numpy.ndarray): The validation target
        
    Returns:
        sklearn.pipeline.Pipeline: The trained model
    """
    print("\n--- Training XGBoost Regression Model ---")
    
    # Create a pipeline with XGBoost regression
    xgb_pipeline = Pipeline([
        ('model', XGBRegressor.XGBRegressor(random_state=42))
    ])
    
    # Define hyperparameter grid
    param_grid = {
        'model__n_estimators': [50, 100, 200],
        'model__max_depth': [3, 6, 9],
        'model__learning_rate': [0.01, 0.1, 0.3]
    }
    
    # Perform grid search with cross-validation
    grid_search = GridSearchCV(
        xgb_pipeline, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    # Get the best model
    best_xgb = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    print(f"Best hyperparameters: {best_params}")
    
    # Evaluate on training set
    y_train_pred = best_xgb.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    
    print(f"Training RMSE: {train_rmse:.4f}")
    print(f"Training R²: {train_r2:.4f}")
    
    # Evaluate on validation set
    y_val_pred = best_xgb.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_r2 = r2_score(y_val, y_val_pred)
    
    print(f"Validation RMSE: {val_rmse:.4f}")
    print(f"Validation R²: {val_r2:.4f}")
    
    return best_xgb, val_rmse, val_r2

def evaluate_on_test_set(models, X_test, y_test):
    """
    Evaluate all models on the test set
    
    Args:
        models (dict): Dictionary of trained models
        X_test (pandas.DataFrame): The test features
        y_test (numpy.ndarray): The test target
    """
    print("\n--- Evaluating Models on Test Set ---")
    
    results = {}
    
    for name, model in models.items():
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2
        }
        
        print(f"\n{name} Test Results:")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R²: {r2:.4f}")
    
    # Create a DataFrame with the results
    results_df = pd.DataFrame({
        model_name: {
            'RMSE': metrics['RMSE'],
            'MAE': metrics['MAE'],
            'R²': metrics['R²']
        } for model_name, metrics in results.items()
    }).T
    
    # Plot the results
    plt.figure(figsize=(12, 6))
    
    # Plot RMSE
    plt.subplot(1, 2, 1)
    results_df['RMSE'].plot(kind='bar')
    plt.title('RMSE by Model')
    plt.ylabel('RMSE')
    plt.grid(axis='y', alpha=0.3)
    
    # Plot R²
    plt.subplot(1, 2, 2)
    results_df['R²'].plot(kind='bar')
    plt.title('R² by Model')
    plt.ylabel('R²')
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    
    # Save the results to a CSV file
    results_df.to_csv('model_results.csv')
    
    return results_df

def plot_feature_importance(models, feature_names):
    """
    Plot feature importance for applicable models
    
    Args:
        models (dict): Dictionary of trained models
        feature_names (list): List of feature names
    """
    print("\n--- Plotting Feature Importance ---")
    
    # For Random Forest
    if 'Random Forest' in models:
        rf_model = models['Random Forest'].named_steps['model']
        
        # Get feature importance
        importances = rf_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        plt.title('Random Forest Feature Importance')
        plt.bar(range(len(importances)), importances[indices], align='center')
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig('rf_feature_importance.png')
        
        # Print top 10 features
        print("\nTop 10 features by importance (Random Forest):")
        for i in range(10):
            print(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
    
    # For XGBoost
    if 'XGBoost' in models:
        xgb_model = models['XGBoost'].named_steps['model']
        
        # Get feature importance
        importances = xgb_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        plt.title('XGBoost Feature Importance')
        plt.bar(range(len(importances)), importances[indices], align='center')
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig('xgb_feature_importance.png')
        
        # Print top 10 features
        print("\nTop 10 features by importance (XGBoost):")
        for i in range(10):
            print(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
    
    # For Linear models (coefficients)
    if 'Linear Regression' in models:
        lr_model = models['Linear Regression'].named_steps['model']
        scaler = models['Linear Regression'].named_steps['scaler']
        
        # Get coefficients
        coefficients = lr_model.coef_
        indices = np.argsort(np.abs(coefficients))[::-1]
        
        # Plot coefficients
        plt.figure(figsize=(12, 8))
        plt.title('Linear Regression Coefficients')
        plt.bar(range(len(coefficients)), coefficients[indices], align='center')
        plt.xticks(range(len(coefficients)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig('lr_coefficients.png')
        
        # Print top 10 features
        print("\nTop 10 features by coefficient magnitude (Linear Regression):")
        for i in range(10):
            print(f"{feature_names[indices[i]]}: {coefficients[indices[i]]:.4f}")

def main():
    """
    Main function to execute Task 3
    """
    # Load the cleaned data
    df_cleaned = load_cleaned_data()
    if df_cleaned is None:
        return
    
    # Prepare data for supervised learning
    X, y, feature_names = prepare_data_for_supervised_learning(df_cleaned)
    
    # Split the data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
    # Train baseline model
    baseline_model, baseline_val_rmse, baseline_val_r2 = train_baseline_model(X_train, y_train, X_val, y_val)
    
    # Train Ridge regression model
    ridge_model, ridge_val_rmse, ridge_val_r2 = train_ridge_regression(X_train, y_train, X_val, y_val)
    
    # Train Random Forest model
    rf_model, rf_val_rmse, rf_val_r2 = train_random_forest(X_train, y_train, X_val, y_val)
    
    # Train XGBoost model
    xgb_model, xgb_val_rmse, xgb_val_r2 = train_xgboost(X_train, y_train, X_val, y_val)
    
    # Collect all models
    models = {
        'Linear Regression': baseline_model,
        'Ridge Regression': ridge_model,
        'Random Forest': rf_model,
        'XGBoost': xgb_model
    }
    
    # Evaluate on test set
    test_results = evaluate_on_test_set(models, X_test, y_test)
    
    # Plot feature importance
    plot_feature_importance(models, feature_names)
    
    # Save the best model
    best_model_name = test_results['R²'].idxmax()
    best_model = models[best_model_name]
    joblib.dump(best_model, 'best_model.pkl')
    
    print(f"\nBest model ({best_model_name}) saved to 'best_model.pkl'")
    print("\nTask 3 completed.")

if __name__ == "__main__":
    main()
