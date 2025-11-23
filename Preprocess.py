import pandas as pd
import re
import spacy
from nltk.corpus import stopwords
import nltk

# Download stopwords if not already
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Load dataset
df = pd.read_csv("Cleaned_Step3_BeyondBlue_NEW.csv")
print(f"Original rows: {len(df)}")

# Drop rows with null in 'Full Text'
df = df.dropna(subset=['Full Text'])


# Preprocessing function
def preprocess(text):
    # Remove URLs
    text = re.sub(r"http\S+|www.\S+", "", text)

    # Lowercase
    text = text.lower()

    # Remove non-alphanumeric (keep emojis)
    text = re.sub(r"[^a-z0-9\s\u263a-\U0001f645]", "", text)

    # spaCy processing
    doc = nlp(text)

    # Lemmatize, remove stopwords and single-character tokens
    tokens = [token.lemma_ for token in doc if
              token.text not in stop_words and len(token.text) > 1 and not token.is_punct and not token.is_space]

    return " ".join(tokens)


# Apply preprocessing
df['Preprocessed_Text'] = df['Full Text'].apply(preprocess)

# Drop rows where preprocessed text is empty
df = df[df['Preprocessed_Text'].str.strip().str.len() > 0]

# Save preprocessed dataset
df.to_csv("Cleaned_Preprocessed.csv", index=False)
print("Preprocessing complete. Saved as Cleaned_Preprocessed.csv")
