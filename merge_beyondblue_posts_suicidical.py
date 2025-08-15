import pandas as pd
import glob
import os

# Set directory where your batch CSVs are stored
directory = "."  # Change this if the files are in a different folder

# Get all matching CSV files
csv_files = sorted(glob.glob(os.path.join(directory, "beyondblue_suicidal_posts_*.csv")))

# Read and concatenate all CSV files
combined_df = pd.concat((pd.read_csv(file) for file in csv_files), ignore_index=True)

# Save to a single merged file
combined_df.to_csv("beyondblue_suicidal_posts_all.csv", index=False)

print(f"✅ Merged {len(csv_files)} files into 'beyondblue_suicidal_posts_all.csv'")
