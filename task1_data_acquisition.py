import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

def load_data():
    print("Loading California Housing dataset...")
    california = fetch_california_housing(as_frame=True)
    df = california.frame
    df['income_category'] = pd.qcut(df['MedInc'], q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    df['age_category'] = pd.cut(df['HouseAge'], bins=[0, 10, 20, 30, 40, np.inf], labels=['New', 'Recent', 'Medium', 'Old', 'Very Old'])
    df['population_density'] = pd.qcut(df['Population'], q=3, labels=['Low', 'Medium', 'High'])
    for col in df.columns[:5]:  
        mask = np.random.random(len(df)) < 0.05
        df.loc[mask, col] = np.nan
    print(f"Dataset loaded with shape: {df.shape}")
    return df

def perform_eda(df):
    print("\n--- Exploratory Data Analysis ---")
    
    print("\nDataset Info:")
    print(df.info())
    
    print("\nSummary Statistics:")
    print(df.describe().round(2))
    
    print("\nMissing Values:")
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Values': missing_values,
        'Percentage': missing_percentage.round(2)
    })
    print(missing_df)
    
    print("\nVisualizing distributions of numerical features...")
    
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(df.select_dtypes(include=np.number).columns):
        plt.subplot(3, 3, i+1)
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f'Distribution of {col}')
        plt.tight_layout()
    plt.savefig('numerical_distributions.png')
    
    print("\nVisualizing relationships between key features...")
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='MedInc', y='MedHouseVal', data=df, alpha=0.5, hue='income_category')
    plt.title('Median Income vs. Median House Value')
    plt.xlabel('Median Income')
    plt.ylabel('Median House Value')
    plt.savefig('income_vs_house_value.png')
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='age_category', y='MedHouseVal', data=df)
    plt.title('House Age vs. Median House Value')
    plt.xlabel('House Age Category')
    plt.ylabel('Median House Value')
    plt.savefig('age_vs_house_value.png')
    
    plt.figure(figsize=(12, 10))
    correlation_matrix = df.select_dtypes(include=np.number).corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    plt.savefig('correlation_matrix.png')
    
    return

def clean_data(df):
    print("\n--- Data Cleaning ---")
    
    missing_values = df.isnull().sum()
    total_missing = missing_values.sum()
    
    if total_missing > 0:
        print("\nDetected missing values in the dataset:")
        print(missing_values[missing_values > 0])
        print(f"Total missing values: {total_missing} ({(total_missing / df.size) * 100:.2f}% of all data)")
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
        plt.title('Missing Values Heatmap')
        plt.tight_layout()
        plt.savefig('missing_values_heatmap.png')
        plt.close()
        print("Missing values heatmap saved to 'missing_values_heatmap.png'")
    else:
        print("\nNo missing values detected in the dataset.")
    
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"\nNumerical columns: {numerical_cols}")
    print(f"Categorical columns: {categorical_cols}")
    
    print("\nSetting up imputation strategies:")
    print("  - For numerical columns: Using median imputation (robust to outliers)")
    print("  - For categorical columns: Using most frequent value imputation")
    
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    df_cleaned = df.copy()
    
    print("\nApplying preprocessing pipeline...")
    preprocessed_data = preprocessor.fit_transform(df_cleaned)
    
    onehot_cols = []
    if categorical_cols:
        onehot_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
        onehot_cols = onehot_encoder.get_feature_names_out(categorical_cols)
    
    all_feature_names = numerical_cols + onehot_cols.tolist()
    df_preprocessed = pd.DataFrame(
        preprocessed_data, 
        columns=all_feature_names,
        index=df_cleaned.index
    )
    
    if df_preprocessed.isnull().sum().sum() > 0:
        print("\nWarning: There are still missing values after preprocessing!")
        print(df_preprocessed.isnull().sum()[df_preprocessed.isnull().sum() > 0])
        df_preprocessed = df_preprocessed.fillna(df_preprocessed.mean())
        print("Applied additional mean imputation to handle remaining missing values.")
    else:
        print("\nVerified: No missing values remain after preprocessing.")
    
    print(f"Cleaned dataset shape: {df_preprocessed.shape}")
    print(f"Number of features after preprocessing: {len(df_preprocessed.columns)}")
    
    print("\nImputation Summary:")
    for col in numerical_cols:
        if df[col].isnull().sum() > 0:
            imputed_value = preprocessor.named_transformers_['num'].named_steps['imputer'].statistics_[numerical_cols.index(col)]
            print(f"  - {col}: {df[col].isnull().sum()} missing values imputed with median {imputed_value:.4f}")
    
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            most_frequent = df[col].mode()[0]
            print(f"  - {col}: {df[col].isnull().sum()} missing values imputed with mode '{most_frequent}'")
    
    return df_cleaned, df_preprocessed, preprocessor

def main():
    df = load_data()
    
    perform_eda(df)
    
    df_cleaned, df_preprocessed, preprocessor = clean_data(df)
    
    df_cleaned.to_csv('california_housing_cleaned.csv', index=False)
    df_preprocessed.to_csv('california_housing_preprocessed.csv', index=False)
    
    with open('data_quality_report.txt', 'w') as f:
        f.write("California Housing Dataset - Data Quality Report\n")
        f.write("==============================================\n\n")
        
        f.write("Original Dataset:\n")
        f.write(f"  - Shape: {df.shape}\n")
        f.write(f"  - Missing values: {df.isnull().sum().sum()}\n")
        f.write(f"  - Duplicated rows: {df.duplicated().sum()}\n\n")
        
        f.write("Cleaned Dataset:\n")
        f.write(f"  - Shape: {df_cleaned.shape}\n")
        f.write(f"  - Missing values: {df_cleaned.isnull().sum().sum()}\n")
        f.write(f"  - Duplicated rows: {df_cleaned.duplicated().sum()}\n\n")
        
        f.write("Preprocessing Steps:\n")
        f.write("  - Missing value imputation (median for numerical, mode for categorical)\n")
        f.write("  - Standardization of numerical features\n")
        f.write("  - One-hot encoding of categorical features\n\n")
        
        f.write("Feature Information:\n")
        f.write(f"  - Original features: {df.columns.tolist()}\n")
        f.write(f"  - Preprocessed features: {len(df_preprocessed.columns)}\n")
    
    print("\nTask 1 completed.")
    print("  - Cleaned data saved to 'california_housing_cleaned.csv'")
    print("  - Preprocessed data saved to 'california_housing_preprocessed.csv'")
    print("  - Data quality report saved to 'data_quality_report.txt'")

if __name__ == "__main__":
    main()
