import pandas as pd

# Load cleaned posts (no URLs)
clean_df = pd.read_csv("beyondblue_depression_posts_all_clean.csv")
clean_df["Post ID"] = clean_df["Post ID"].astype(str)

# Load the URLs mapping
urls_df = pd.read_csv("post_urls.csv")
urls_df["Post ID"] = urls_df["Post ID"].astype(str)

# Merge
merged_df = clean_df.merge(urls_df, on="Post ID", how="left")

# Save
merged_df.to_csv("beyondblue_depression_posts_with_urls.csv", index=False)

print("✅ Created beyondblue_depression_posts_with_urls.csv")
