import pandas as pd

# Load your dataframe
df = pd.read_csv('Final_Objective1_Dataset_With_VADER.csv')

# Function to extract and pad top 5 emotions
def extract_top5_emotions(emotion_str):
    if pd.isna(emotion_str):
        return ['no_emotion'] * 5
    emotions = [e.strip() for e in emotion_str.split(',')]
    while len(emotions) < 5:
        emotions.append('no_emotion')
    return emotions[:5]

# Apply the function and create 5 new columns
emotion_cols = df['Emotions'].apply(extract_top5_emotions)
df[['top_emotion_1', 'top_emotion_2', 'top_emotion_3', 'top_emotion_4', 'top_emotion_5']] = pd.DataFrame(emotion_cols.tolist(), index=df.index)
