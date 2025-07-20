# Data Science Project

This repository contains a comprehensive data science project that demonstrates various techniques in data analysis, machine learning, and SQL.

## Project Structure

- `zeidex_tasks.ipynb`: Jupyter notebook containing Tasks 1-3 (Data Acquisition & Cleanup, Unsupervised Learning, and Supervised Learning)
- `sql_challenge/`: Directory containing SQL scripts for Task 4
  - `setup.sql`: Script to create the database and load the data
  - `queries.sql`: SQL queries to answer the required questions
- `requirements.txt`: List of Python dependencies

## Approach

1. **Data Selection**: I've chosen the California Housing dataset, which contains information about housing in California districts. This dataset has over 20,000 rows and features both numerical and categorical variables.

2. **Data Cleaning & EDA**: The notebook includes comprehensive exploratory data analysis, handling of missing values, encoding of categorical variables, and normalization of numerical features.

3. **Unsupervised Learning**: I've implemented K-Means and DBSCAN clustering algorithms to identify patterns in the housing data, with visualizations using PCA and t-SNE.

4. **Supervised Learning**: For the supervised task, I've predicted median house values using various models including Linear Regression, Random Forest, and XGBoost, with proper train/validation/test splits and hyperparameter tuning.

5. **SQL Challenge**: The SQL scripts demonstrate how to load the cleaned dataset into a SQLite database and perform various queries including aggregations, rankings, window functions, and joins.

## Assumptions

- The dataset is assumed to be relatively clean but may contain some missing values.
- For clustering, we assume that geographical proximity and similar housing characteristics indicate similar housing markets.
- For the supervised task, we assume that the features provided are sufficient to predict median house values.

## Instructions to Run

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the Jupyter notebook:
   ```
   jupyter notebook zeidex_tasks.ipynb
   ```

3. For the SQL challenge, you can run the SQL scripts using any SQL client that supports SQLite, or use the Python sqlite3 module as demonstrated in the notebook.

4. Ouput folder contain graphs and model performance