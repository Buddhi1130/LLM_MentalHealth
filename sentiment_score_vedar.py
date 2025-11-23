import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')

# Load your dataset
df = pd.read_csv("Final_Objective1_Dataset.csv")

# Initialize VADER
sid = SentimentIntensityAnalyzer()

# Apply to your preprocessed column
sentiment_scores = df['Preprocessed_Text'].apply(sid.polarity_scores)

# Convert dict into separate columns
sentiment_df = sentiment_scores.apply(pd.Series)

# Merge with original dataframe
df = pd.concat([df, sentiment_df], axis=1)

# Save new dataset
df.to_csv("Final_Objective1_Dataset_With_VADER.csv", index=False)
print("✅ VADER Sentiment scores added!")
