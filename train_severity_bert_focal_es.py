# -*- coding: utf-8 -*-
import os, numpy as np, pandas as pd, torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)

# ---------------- Config ----------------
CSV_PATH  = "Objective1_Merged_CLEAN.csv"
TEXT_COL  = "Text_With_EmojiDesc"
LABEL_COL = "Severity"

MODEL_ID  = "microsoft/deberta-v3-base"
OUT_DIR   = "./severity-bert-focal-es"

RANDOM_SEED = 42
NUM_EPOCHS  = 8
LR          = 2e-5
TRAIN_BS    = 4
EVAL_BS     = 4
MAX_LEN     = 256

WARMUP_RATIO = 0.06
WEIGHT_DECAY = 0.01
GRAD_ACCUM   = 8
PATIENCE     = 2     # early stopping patience (epochs)

# -------------- Data --------------------
df = pd.read_csv(CSV_PATH)
df = df[[TEXT_COL, LABEL_COL]].dropna().reset_index(drop=True)
df[TEXT_COL] = df[TEXT_COL].astype(str).str.strip()
df = df[df[TEXT_COL].str.len() > 0].reset_index(drop=True)

label_list = ["mild_distress","moderate_distress","severe_crisis"]
label2id = {l:i for i,l in enumerate(label_list)}
id2label = {i:l for l,i in label2id.items()}
df["labels"] = df[LABEL_COL].map(label2id).astype(int)

train_df, temp_df = train_test_split(
    df, test_size=0.20, random_state=RANDOM_SEED, stratify=df["labels"]
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, random_state=RANDOM_SEED, stratify=temp_df["labels"]
)

train_ds = Dataset.from_pandas(train_df[[TEXT_COL,"labels"]], preserve_index=False)
val_ds   = Dataset.from_pandas(val_df[[TEXT_COL,"labels"]],   preserve_index=False)
test_ds  = Dataset.from_pandas(test_df[[TEXT_COL,"labels"]],  preserve_index=False)
ds = DatasetDict({"train":train_ds, "validation":val_ds, "test":test_ds})

# -------------- Tokenizer ---------------
tok = AutoTokenizer.from_pretrained(MODEL_ID)
def tokenize(batch):
    return tok(batch[TEXT_COL], truncation=True, padding="max_length", max_length=MAX_LEN)
ds = ds.map(tokenize, batched=True, remove_columns=[TEXT_COL])
ds.set_format("torch", columns=["input_ids","attention_mask","labels"])

# -------------- Model -------------------
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID, num_labels=3, id2label=id2label, label2id=label2id
)
model.gradient_checkpointing_enable()
# --- put this ABOVE the Trainer creation ---
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

# class weights (imbalance) + focal loss
train_counts = train_df["labels"].value_counts().sort_index().values.astype(np.float32)
class_weights = (train_counts.sum() / (len(train_counts) * train_counts))
class_weights = torch.tensor(class_weights, dtype=torch.float)

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        # Register alpha as a buffer so .to(device) moves it (important for MPS/CPU/GPU)
        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, target):
        # cross_entropy expects target Long dtype
        target = target.long()
        ce = torch.nn.functional.cross_entropy(
            logits, target, weight=self.alpha, reduction="none"
        )
        pt = torch.exp(-ce)                 # pt = prob of the true class
        loss = ((1 - pt) ** self.gamma) * ce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

class FocalTrainer(Trainer):
    def __init__(self, *args, focal_alpha=None, focal_gamma=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )
        logits = outputs.logits
        # Ensure focal (and its buffer alpha) are on the same device as logits
        self.focal = self.focal.to(logits.device)
        # Also move labels to the same device
        labels = labels.to(logits.device)
        loss = self.focal(logits, labels)
        return (loss, outputs) if return_outputs else loss


# -------------- Training args -----------
args = TrainingArguments(
    output_dir=OUT_DIR,
    seed=RANDOM_SEED,
    learning_rate=LR,
    per_device_train_batch_size=TRAIN_BS,
    per_device_eval_batch_size=EVAL_BS,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=WEIGHT_DECAY,
    warmup_ratio=WARMUP_RATIO,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    logging_steps=50,
    report_to="none",
    dataloader_pin_memory=False,
    dataloader_num_workers=0,
    gradient_checkpointing=True,   # << important
)


trainer = FocalTrainer(
    model=model,
    args=args,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    tokenizer=tok,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)],
    focal_alpha=class_weights,
    focal_gamma=2.0,
)
# right before trainer.train()
import torch
if torch.backends.mps.is_available():
    torch.mps.empty_cache()

trainer.train()

# Test
metrics = trainer.evaluate(ds["test"])
print("\nTest metrics:", metrics)

# Save artifacts
os.makedirs(os.path.join(OUT_DIR,"best"), exist_ok=True)
trainer.model.save_pretrained(os.path.join(OUT_DIR,"best"))
tok.save_pretrained(os.path.join(OUT_DIR,"best"))

# Confusion matrix + per-row predictions
pred_logits = trainer.predict(ds["test"]).predictions
y_pred = pred_logits.argmax(axis=-1)
y_true = test_df["labels"].to_numpy()
cm = confusion_matrix(y_true, y_pred)
pd.DataFrame(cm, index=label_list, columns=label_list).to_csv(os.path.join(OUT_DIR,"confusion_matrix_test.csv"))

test_dump = test_df.copy()
test_dump["pred_id"] = y_pred
test_dump["pred"] = test_dump["pred_id"].map({i:l for i,l in enumerate(label_list)})
test_dump.to_csv(os.path.join(OUT_DIR,"test_predictions.csv"), index=False)
print("Saved model + metrics to:", OUT_DIR)
