# ---- GoEmotion_probabilities.py  (token-hardened version) ----
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline


MY_READ_TOKEN = "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"
INPUT  = "Final_Objective1_With_VADER_comp.csv"
OUTPUT = "Final_with_GoEmotions_Probs.csv"
TEXT_COL = "Text_With_EmojiDesc"
MODEL_ID = "SamLowe/roberta-base-go_emotions"

# =======================

# 1) Make sure the token is visible to both huggingface_hub and transformers
os.environ["HUGGINGFACE_HUB_TOKEN"] = MY_READ_TOKEN
os.environ["HF_TOKEN"] = MY_READ_TOKEN
try:
    login(token=MY_READ_TOKEN, add_to_git_credential=False)
except Exception:
    # If already logged in, this will just pass
    pass

# 2) Load tokenizer/model EXPLICITLY with token (works on old/new transformers)
#    Try the new kwarg first; fall back to the legacy one if needed.
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=MY_READ_TOKEN)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, token=MY_READ_TOKEN)
except TypeError:
    # Older transformers use `use_auth_token`
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_auth_token=MY_READ_TOKEN)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, use_auth_token=MY_READ_TOKEN)

clf = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    return_all_scores=True,
    truncation=True,
    top_k=None,
    # device=-1,  # uncomment to force CPU
)

# 3) Read data
df = pd.read_csv(INPUT)

labels_order = None
probs_matrix = []
top5_labels = []
top5_scores = []

BATCH = 32
for i in tqdm(range(0, len(df), BATCH)):
    texts = df[TEXT_COL].astype(str).iloc[i:i+BATCH].tolist()
    outs = clf(texts)  # list of lists of {label, score}

    for out in outs:
        out_sorted = sorted(out, key=lambda x: x["score"], reverse=True)
        if labels_order is None:
            labels_order = [o["label"] for o in out_sorted]  # lock a consistent label order

        # full 28-dim vector in fixed order
        by_label = {o["label"]: o["score"] for o in out}
        vec = [by_label[l] for l in labels_order]
        probs_matrix.append(vec)

        # top-5 lists
        top5_labels.append([o["label"] for o in out_sorted[:5]])
        top5_scores.append([o["score"] for o in out_sorted[:5]])

# 4) Attach to dataframe and save
probs = np.vstack(probs_matrix)
for j, lab in enumerate(labels_order):
    df[f"goem_{lab}"] = probs[:, j]

df["goem_top5_labels"] = top5_labels
df["goem_top5_scores"] = top5_scores

df.to_csv(OUTPUT, index=False)
print(f"Saved: {OUTPUT}")
print("First 5 prob columns:", [f"goem_{l}" for l in labels_order[:5]])
