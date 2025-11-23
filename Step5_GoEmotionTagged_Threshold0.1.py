import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load dataset
df = pd.read_csv("Cleaned_Step4_Tokenized.csv")

# Load model & tokenizer (corrected model)
model_name = "bhadresh-savani/bert-base-go-emotion"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# GoEmotions labels (28 + neutral)
emotion_labels = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

# Emotion prediction function with 0.1 threshold
def predict_emotions(text, threshold=0.1):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        scores = torch.sigmoid(outputs.logits)[0].cpu().numpy()

    predicted_emotions = [emotion_labels[i] for i, score in enumerate(scores) if score >= threshold]
    return predicted_emotions

# Apply to data
df['Predicted Emotions'] = df['Full Text'].apply(lambda x: predict_emotions(str(x), threshold=0.1))

# Save output
df.to_csv("Cleaned_Step5_GoEmotionTagged_Threshold0.1.csv", index=False)
print("✅ Saved: Cleaned_Step5_GoEmotionTagged_Threshold0.1.csv")
