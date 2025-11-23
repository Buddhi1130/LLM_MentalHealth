import pandas as pd
import glob

# Get all scraped CSV files
files = sorted(glob.glob("beyondblue_depression_posts_*.csv"))

print(f"Found {len(files)} files to merge.")

# Read them all into DataFrames
dfs = [pd.read_csv(f) for f in files]

# Concatenate
merged_df = pd.concat(dfs, ignore_index=True)

# Save the merged CSV
merged_df.to_csv("beyondblue_depression_posts_all.csv", index=False)

print(" All CSVs merged into beyondblue_depression_posts_all.csv")
