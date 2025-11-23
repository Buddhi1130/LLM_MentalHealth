import pandas as pd
from transformers import pipeline
from tqdm import tqdm

# Load your full cleaned/tokenized dataset
df = pd.read_csv("Cleaned_Step4_Tokenized.csv")

# Convert token list string to actual list if needed
import ast
df['Tokens'] = df['Tokens'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# Re-join tokens into text for emotion analysis
df['Rejoined_Text'] = df['Tokens'].apply(lambda tokens: " ".join(tokens))

# Load GoEmotions 28-emotion classifier
emotion_classifier = pipeline("text-classification",
                              model="bhadresh-savani/distilbert-base-uncased-emotion",
                              return_all_scores=True)

# Define 28 emotion labels
emotion_labels = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

# Emotion prediction function with threshold
def predict_emotions(text, threshold=0.3):
    try:
        results = emotion_classifier(text)
        return [res['label'] for res in results[0] if res['score'] >= threshold]
    except Exception as e:
        return []

# Apply with progress bar
tqdm.pandas(desc="Predicting emotions")
df["Predicted Emotions"] = df["Rejoined_Text"].progress_apply(predict_emotions)

# Drop temporary column
df = df.drop(columns=["Rejoined_Text"])

# Save the final file
df.to_csv("Cleaned_Step5_GoEmotionTagged.csv", index=False)
print("✅ Step 5 complete. File saved as Cleaned_Step5_GoEmotionTagged.csv")
