from textblob import TextBlob
import pandas as pd

# Load your data
df = pd.read_csv("Final_Objective1_Dataset.csv")

# Define function to extract sentiment polarity and subjectivity
def get_sentiment(text):
    blob = TextBlob(text)
    return pd.Series([blob.sentiment.polarity, blob.sentiment.subjectivity])

# Apply to your preprocessed text
df[['Polarity', 'Subjectivity']] = df['Preprocessed_Text'].apply(get_sentiment)

# Save the new version
df.to_csv("Final_Objective1_WithSentiment.csv", index=False)

print("Sentiment scores added and file saved as Final_Objective1_WithSentiment.csv")
