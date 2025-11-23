import pandas as pd

# Load preprocessed data
df = pd.read_csv("Cleaned_Preprocessed.csv")

# View first few rows
print(df.head())

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load MentalBERT model and tokenizer
model_name = "mental/mental-bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()  # set to evaluation mode

# Select sample (you can increase later)
sample_df = df.sample(100, random_state=42).copy()

# Inference function
def classify_post(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
    return predicted_class

# Apply to preprocessed column
sample_df['MentalBERT_Label'] = sample_df['Preprocessed_Text'].apply(classify_post)

# Preview
print(sample_df[['Preprocessed_Text', 'MentalBERT_Label']].head())

label_map = {
    0: 'depression',
    1: 'anxiety',
    2: 'suicidal_ideation',
    3: 'ptsd',
    4: 'ocd',
    5: 'neutral'
}

sample_df['MentalBERT_Label_Name'] = sample_df['MentalBERT_Label'].map(label_map)

sample_df.to_csv("MentalBERT_Output_Sample.csv", index=False)
print("Saved MentalBERT results.")
