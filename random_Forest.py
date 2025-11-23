# random_forest_severity.py
# -------------------------------------------------------------
# Train a Random Forest to predict severity (mild/moderate/severe)
# from GoEmotions probabilities (+ VADER compound [+ optional emoji buckets])
# -------------------------------------------------------------

import os
import re
import ast
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import seaborn as sns

# Optional balancing (toggle with USE_SMOTE)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# -----------------------------
# Config
# -----------------------------
CSV_PATH = "Objective1_Merged_CLEAN.csv"      #
USE_EMOJI_BUCKETS = True                      #
USE_SMOTE = True                              #
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_ESTIMATORS = 600
MAX_DEPTH = None

# -----------------------------
# 1) Load data
# -----------------------------
df = pd.read_csv(CSV_PATH)

# Safety check: required columns
required_cols = [
    "Severity",
    "vader_compound_with_emojis",
    "Emoji_Descriptions",
]
# GoEmotions probability columns (28 labels)
GOEM_LABELS = [
    'admiration','amusement','anger','annoyance','approval','caring','confusion','curiosity',
    'desire','disappointment','disapproval','disgust','embarrassment','excitement','fear',
    'gratitude','grief','joy','love','nervousness','optimism','pride','realization','relief',
    'remorse','sadness','surprise','neutral'
]
goem_cols = [f"goem_{lab}" for lab in GOEM_LABELS]
required_cols.extend(goem_cols)

missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing expected columns: {missing}")

# Keep only essentials (and drop NA rows just in case)
df = df.dropna(subset=["Severity", "vader_compound_with_emojis"] + goem_cols).reset_index(drop=True)

# -----------------------------
# 2) Optional: simple emoji buckets from descriptors
# -----------------------------
def emoji_buckets(desc: str) -> dict:
    s = str(desc).lower()
    return {
        "emoji_cry":        int(bool(re.search(r"\bcry|tears|sob|loudly crying|sad face", s))),
        "emoji_sad":        int(bool(re.search(r"\bsad|pensive|weary\b", s))),
        "emoji_heartbreak": int(bool(re.search(r"broken heart|heartbreak", s))),
        "emoji_fear":       int(bool(re.search(r"\bfear|scream|cold sweat\b", s))),
        "emoji_anger":      int(bool(re.search(r"\bangry|rage|pouting\b", s))),
        "emoji_joy":        int(bool(re.search(r"\blaugh|joy|grin|smile\b", s))),
        "emoji_love":       int(bool(re.search(r"\bheart\b", s))),
    }

emoji_df = None
if USE_EMOJI_BUCKETS:
    emoji_df = df["Emoji_Descriptions"].apply(emoji_buckets).apply(pd.Series)
else:
    emoji_df = pd.DataFrame(index=df.index)

# -----------------------------
# 3) Build feature matrix X, target y
# -----------------------------
num_feats = df[goem_cols + ["vader_compound_with_emojis"]].copy()

# Feature matrix
X = pd.concat([num_feats, emoji_df], axis=1)

# Target (ensure consistent label names)
df["Severity"] = df["Severity"].astype(str).str.strip()
y = df["Severity"]

print("Feature matrix shape:", X.shape)
print("Targets distribution (%):")
print(y.value_counts(normalize=True) * 100)
print()

# -----------------------------
# 4) Train/test split (stratified)
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# -----------------------------
# 5) Model: Random Forest (+ optional SMOTE)
# -----------------------------
rf = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    max_depth=MAX_DEPTH,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    class_weight=None  # let SMOTE handle balancing if used
)

if USE_SMOTE:
    # Only oversample training set inside the pipeline
    model = ImbPipeline(steps=[
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("rf", rf)
    ])
else:
    model = rf

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# -----------------------------
# 6) Evaluation
# -----------------------------
print("\nRandom Forest (GoEmotions + VADER{}):\n".format(" + EmojiBuckets" if USE_EMOJI_BUCKETS else ""))
print(classification_report(y_test, y_pred, digits=3))

# -----------------------------
# 7) Confusion matrix (saved)
# -----------------------------
cm = confusion_matrix(y_test, y_pred, labels=sorted(y.unique()))
cm_df = pd.DataFrame(cm, index=sorted(y.unique()), columns=sorted(y.unique()))

plt.figure(figsize=(6, 5))
sns.heatmap(cm_df, annot=True, fmt='d', cmap="Blues")
plt.title("Confusion Matrix — Random Forest")
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.tight_layout()
cm_path = "rf_confusion_matrix.png"
plt.savefig(cm_path, dpi=200)
plt.close()
print(f"Saved confusion matrix to {cm_path}")

# -----------------------------
# 8) Feature importances (saved)
# -----------------------------
# If SMOTE pipeline used, the final estimator is at step 'rf'
rf_est = model.named_steps["rf"] if USE_SMOTE else model
importances = rf_est.feature_importances_
feat_names = list(X.columns)

fi = pd.DataFrame({"feature": feat_names, "importance": importances}).sort_values(
    "importance", ascending=False
)
fi_path = "rf_feature_importances.csv"
fi.to_csv(fi_path, index=False)
print(f"Saved feature importances to {fi_path}")

# Show top 15 in console
print("\nTop 15 features:")
print(fi.head(15).to_string(index=False))
