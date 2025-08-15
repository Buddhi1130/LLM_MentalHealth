import pandas as pd

# Load threads CSV
threads_df = pd.read_csv("beyondblue_depression_threads_all.csv")

# Extract Post ID 
threads_df["Post ID"] = threads_df["link"].apply(lambda x: x.split("/")[-1])

# Make sure Post ID is string
threads_df["Post ID"] = threads_df["Post ID"].astype(str)


threads_df["Post URL"] = threads_df["link"]

# Keep only Post ID and Post URL
urls_df = threads_df[["Post ID", "Post URL"]]

# Save to new CSV
urls_df.to_csv("post_urls.csv", index=False)

print(" Created post_urls.csv with Post IDs and URLs.")
