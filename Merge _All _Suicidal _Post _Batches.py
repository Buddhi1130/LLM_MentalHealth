import pandas as pd
import glob
import os

# Folder where batch files are saved
folder_path = "suicidal_posts_batches"

# Match all CSV files in that folder
csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

# Load and combine all files
all_batches = []
for file in csv_files:
    print(f"🔄 Merging: {file}")
    df = pd.read_csv(file)
    all_batches.append(df)

# Concatenate into one DataFrame
merged_df = pd.concat(all_batches, ignore_index=True)

# Save to final CSV
merged_df.to_csv("beyondblue_suicidal_posts_all.csv", index=False)
print("\n✅ All batches merged successfully into 'beyondblue_suicidal_posts_all.csv'")
