# train_severity_bert.py
# Fine-tune mental/mental-bert-base-uncased for 3-way severity classification


import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

# ---------------------------
# 0) Config
# ---------------------------
CSV = "Objective1_Merged_CLEAN.csv"
TEXT_COL_CANDIDATES = ["Text_With_EmojiDesc"]  # we'll pick the first that exists
LABEL_COL = "Severity"                # values: mild_distress / moderate_distress / severe_crisis
MAX_LEN = 512                        # try 384 or 512 later
MODEL_NAME = "mental/mental-bert-base-uncased"
OUT_DIR = "severity-mentalbert_NEW"

label2id = {"mild_distress": 0, "moderate_distress": 1, "severe_crisis": 2}
id2label = {v: k for k, v in label2id.items()}

# ---------------------------
# 1) Load & split data
# ---------------------------
df = pd.read_csv(CSV)

# pick the best available text column
text_col = None
for c in TEXT_COL_CANDIDATES:
    if c in df.columns:
        text_col = c
        break
if text_col is None:
    raise ValueError(f"None of {TEXT_COL_CANDIDATES} found in CSV columns: {list(df.columns)}")

# keep only what we need; basic filters
df = df[[text_col, LABEL_COL]].dropna()
df = df[df[text_col].str.len() > 10].reset_index(drop=True)

# map labels to ids
df["label"] = df[LABEL_COL].map(label2id)
if df["label"].isna().any():
    bad = df[df["label"].isna()][LABEL_COL].unique()
    raise ValueError(f"Found unexpected label values: {bad}. Expected {list(label2id.keys())}")

# stratified split: train/val/test = 80/10/10
train_df, temp_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["label"]
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, random_state=42, stratify=temp_df["label"]
)

print("\nLabel distribution (train):")
print(train_df["label"].value_counts(normalize=True).sort_index().rename(index=id2label).round(3))
print("\nLabel distribution (val):")
print(val_df["label"].value_counts(normalize=True).sort_index().rename(index=id2label).round(3))
print("\nLabel distribution (test):")
print(test_df["label"].value_counts(normalize=True).sort_index().rename(index=id2label).round(3))

# ---------------------------
# 2) HF datasets + tokenizer
# ---------------------------
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

def tokenize_function(batch):
    return tokenizer(
        batch[text_col],
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN
    )

ds = DatasetDict({
    "train":      Dataset.from_pandas(train_df[[text_col, "label"]]),
    "validation": Dataset.from_pandas(val_df[[text_col, "label"]]),
    "test":       Dataset.from_pandas(test_df[[text_col, "label"]]),
})
ds = ds.map(tokenize_function, batched=True, remove_columns=[text_col])
ds = ds.rename_column("label", "labels")
ds.set_format(type="torch")

# ---------------------------
# 3) Class weights (for imbalance)
# ---------------------------
train_labels_np = np.array(train_df["label"])
counts = np.bincount(train_labels_np, minlength=3)
class_weights = (1.0 / counts) * (len(train_labels_np) / 3.0)
class_weights = torch.tensor(class_weights, dtype=torch.float)
print("\nClass counts:", counts, " => class weights:", class_weights.tolist())

# ---------------------------
# 4) Model, metrics, Trainer
# ---------------------------
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
import evaluate

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,
    id2label=id2label,
    label2id=label2id
)

acc = evaluate.load("accuracy")
f1 = evaluate.load("f1")
prec = evaluate.load("precision")
rec = evaluate.load("recall")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)
    return {
        "accuracy": acc.compute(predictions=preds, references=labels)["accuracy"],
        "f1_macro": f1.compute(average="macro", predictions=preds, references=labels)["f1"],
        "precision_macro": prec.compute(average="macro", predictions=preds, references=labels)["precision"],
        "recall_macro": rec.compute(average="macro", predictions=preds, references=labels)["recall"],
    }

from torch.nn import CrossEntropyLoss

from transformers import Trainer
import torch
import torch.nn.functional as F

class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights  # tensor on correct device in training_step

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Accept **kwargs to be forward-compatible with Trainer calls like
        compute_loss(..., num_items_in_batch=...).
        """
        labels = inputs.get("labels")
        # Forward pass
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.logits

        # Cross-entropy with optional class weights
        # Move weights to same device as logits
        if self.class_weights is not None:
            loss = F.cross_entropy(logits, labels, weight=self.class_weights.to(logits.device))
        else:
            loss = F.cross_entropy(logits, labels)

        return (loss, outputs) if return_outputs else loss

# Training args (tune later)
args = TrainingArguments(
    output_dir=OUT_DIR,
    learning_rate=2e-5,                 # try 1e-5 .. 3e-5
    per_device_train_batch_size=16,     # raise if GPU memory allows
    per_device_eval_batch_size=32,
    num_train_epochs=4,                 # try 3..6
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    logging_steps=50,
    save_total_limit=2,
    report_to="none",
    fp16=torch.cuda.is_available(),     # mixed precision if GPU supports
)

trainer = WeightedTrainer(
    model=model,
    args=args,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    class_weights=class_weights,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# ---------------------------
# 5) Train
# ---------------------------
train_result = trainer.train()
print("\nBest checkpoint:", trainer.state.best_model_checkpoint)

# ---------------------------
# 6) Evaluate on TEST
# ---------------------------
print("\nEvaluating on test set...")
test_metrics = trainer.evaluate(ds["test"])
print("Test metrics:", test_metrics)

# Detailed report + confusion matrix
preds = trainer.predict(ds["test"])
y_true = preds.label_ids
y_pred = np.argmax(preds.predictions, axis=1)

print("\nClassification report (test):")
print(classification_report(y_true, y_pred, target_names=[id2label[i] for i in range(3)]))

cm = confusion_matrix(y_true, y_pred)
cm_df = pd.DataFrame(cm, index=[f"true_{id2label[i]}" for i in range(3)],
                        columns=[f"pred_{id2label[i]}" for i in range(3)])
os.makedirs(OUT_DIR, exist_ok=True)
cm_df.to_csv(os.path.join(OUT_DIR, "confusion_matrix_test.csv"), index=True)
print(f"Saved confusion matrix to {os.path.join(OUT_DIR, 'confusion_matrix_test.csv')}")

# ---------------------------
# 7) (Optional) Save label map & a tiny inference helper
# ---------------------------
with open(os.path.join(OUT_DIR, "label_map.txt"), "w") as f:
    for k, v in label2id.items():
        f.write(f"{k}\t{v}\n")

def predict_texts(texts):
    model.eval()
    enc = tokenizer(texts, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt")
    for k in enc:
        enc[k] = enc[k].to(model.device)
    with torch.no_grad():
        out = model(**enc).logits
        probs = torch.softmax(out, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)
    return [id2label[i] for i in preds], probs

# Quick smoke test for inference
sample_texts = [
    "I feel overwhelmed and can’t see a way out. Please help.",
    "Had a rough day but I think I’ll be okay after some rest.",
    "My anxiety flares up sometimes, but I can manage."
]
pred_labels, pred_probs = predict_texts(sample_texts)
for t, lab, pr in zip(sample_texts, pred_labels, pred_probs):
    print(f"\nTEXT: {t}\nPRED: {lab}\nPROBS: mild={pr[0]:.3f}, moderate={pr[1]:.3f}, severe={pr[2]:.3f}")

from transformers import BertTokenizerFast

# Define where to save
save_dir = "./severity_bert_model_NEW"

# Save model, tokenizer, and training arguments
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

print(f"Model saved to {save_dir}")
