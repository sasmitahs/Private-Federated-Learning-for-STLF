import pandas as pd
import os

# Define the path to the directory containing the CSV files
base_dir = '/home/user/DPFL-Sasmita/FL-Baseline-Codes/ashrae-energy-prediction'

# Initialize an empty list to store dataframes
df_list = []

# Iterate over each file in the directory
for file in os.listdir(base_dir):
    if file.endswith('.csv'):
        file_path = os.path.join(base_dir, file)
        try:
            # Read the CSV file into a dataframe
            df = pd.read_csv(file_path)
            
            # Check if the dataframe is empty
            if df.empty:
                print(f"Skipping empty file: {file_path}")
                continue
            
            # Drop rows where all elements are NaN
            df.dropna(how='all', inplace=True)
            
            # Check if the dataframe is still empty after dropping NaNs
            if df.empty:
                print(f"File after dropping NaNs is empty: {file_path}")
                continue
            
            # Append the dataframe to the list
            df_list.append(df)
            print(f"Processed file: {file_path}")
        
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

# Check if df_list is empty before attempting to concatenate
if df_list:
    # Concatenate all dataframes into a single dataframe
    combined_df = pd.concat(df_list, ignore_index=True)

    # Perform any additional preprocessing steps as needed
    # For example, converting timestamp to datetime
    if 'timestamp' in combined_df.columns:
        combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])

    # Save the combined dataframe to a new CSV file
    combined_df.to_csv('combined_preprocessed_data.csv', index=False)
    print("Data preprocessing complete. Combined file saved as 'combined_preprocessed_data.csv'.")
else:
    print("No valid data files found. No data to process.")
