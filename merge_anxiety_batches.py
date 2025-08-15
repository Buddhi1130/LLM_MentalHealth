import pandas as pd
import glob

# Glob pattern to find all batch files
file_pattern = "beyondblue_anxiety_posts_*.csv"

# Find all matching files
csv_files = sorted(glob.glob(file_pattern))

print(f" Found {len(csv_files)} batch files.")

# Read and concatenate
all_dfs = []
for file in csv_files:
    print(f"🔹 Loading: {file}")
    df = pd.read_csv(file)
    all_dfs.append(df)

merged_df = pd.concat(all_dfs, ignore_index=True)

print(f"\n Merged DataFrame shape: {merged_df.shape}")

# Save merged CSV
output_file = "beyondblue_anxiety_posts_all_merged.csv"
merged_df.to_csv(output_file, index=False)

print(f"\n All batches merged and saved to {output_file}")
