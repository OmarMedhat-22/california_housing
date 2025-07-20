#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main script to run all tasks sequentially
"""

import os
import sys
import subprocess
import time

def run_task(task_script, description):
    """
    Run a task script and print its output
    
    Args:
        task_script (str): Path to the task script
        description (str): Description of the task
    """
    print(f"\n{'='*80}")
    print(f"Running {description}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(['python', task_script], check=True)
        if result.returncode == 0:
            print(f"\n{description} completed successfully.")
        else:
            print(f"\nError: {description} failed with return code {result.returncode}")
            sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"\nError: {description} failed with return code {e.returncode}")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {description} failed with exception: {str(e)}")
        sys.exit(1)
    
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")

def main():
    """
    Main function to run all tasks
    """
    print("Starting Data Science Project")
    
    # Task 1: Data Acquisition & Cleanup
    run_task('task1_data_acquisition.py', 'Task 1: Data Acquisition & Cleanup')
    
    # Task 2: Unsupervised Learning
    run_task('task2_unsupervised_learning.py', 'Task 2: Unsupervised Learning')
    
    # Task 3: Supervised Learning
    run_task('task3_supervised_learning.py', 'Task 3: Supervised Learning')
    
    # Task 4: SQL Challenge
    print(f"\n{'='*80}")
    print(f"Running Task 4: SQL Challenge")
    print(f"{'='*80}\n")
    
    try:
        os.chdir('sql_challenge')
        result = subprocess.run(['python', 'load_data.py'], check=True)
        if result.returncode == 0:
            print("\nSQL data loading completed successfully.")
        else:
            print(f"\nError: SQL data loading failed with return code {result.returncode}")
        os.chdir('..')
    except subprocess.CalledProcessError as e:
        print(f"\nError: SQL Challenge failed with return code {e.returncode}")
    except Exception as e:
        print(f"\nError: SQL Challenge failed with exception: {str(e)}")
    
    print("\nAll tasks completed. Check the generated files and results.")

if __name__ == "__main__":
    main()
