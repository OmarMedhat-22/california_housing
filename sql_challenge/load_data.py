#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sqlite3
import pandas as pd
import os
import sys

def load_data_to_sqlite():
    print("Loading data into SQLite database...")
    
    if not os.path.exists('../california_housing_cleaned.csv'):
        print("Error: Cleaned data file not found. Please run Task 1 first.")
        sys.exit(1)
    
    df = pd.read_csv('../california_housing_cleaned.csv')
    print(f"Loaded data with shape: {df.shape}")
    
    conn = sqlite3.connect('california_housing.db')
    cursor = conn.cursor()
    
    with open('setup.sql', 'r') as f:
        setup_script = f.read()
        conn.executescript(setup_script)
    
    print("Inserting data into the housing table...")
    df.to_sql('housing', conn, if_exists='replace', index=True, index_label='id')
    
    cursor.execute("SELECT COUNT(*) FROM housing")
    count = cursor.fetchone()[0]
    print(f"Successfully loaded {count} rows into the housing table")
    
    cursor.execute("SELECT COUNT(*) FROM region_metadata")
    count = cursor.fetchone()[0]
    print(f"Region metadata table contains {count} rows")
    
    conn.close()
    print("Data loading completed successfully.")

if __name__ == "__main__":
    load_data_to_sqlite()
