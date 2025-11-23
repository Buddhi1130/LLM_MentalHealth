# -*- coding: utf-8 -*-
"""Analysis_I.ipynb


# --- Step 2: Analyze Category vs Severity Distribution ---

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#  Load the balanced dataset
df = pd.read_csv("Objective1_with_VA_balanced.csv")

# Check available columns
print("Columns:", df.columns.tolist())

#  Standardize key column names
df['Severity'] = df['Severity'].str.strip().str.lower()
df['Post_Category'] = df['Post_Category'].str.strip().str.title()

# Create a crosstab for percentages
ct = pd.crosstab(df['Post_Category'], df['Severity'], normalize='index') * 100
ct = ct[['mild_distress', 'moderate_distress', 'severe_crisis']]  # order columns
print("\n--- Category vs Severity (% Distribution) ---\n")
print(ct.round(2))

# Save the table
ct.to_csv("Category_Severity_Distribution.csv")

# Visualization – Heatmap
plt.figure(figsize=(8,5))
sns.heatmap(ct, annot=True, cmap="RdYlGn_r", fmt=".1f", cbar_kws={'label': 'Percentage (%)'})
plt.title("Distribution of Severity across Post Categories")
plt.ylabel("Forum Category")
plt.xlabel("Severity Level")
plt.tight_layout()
plt.savefig("Category_Severity_Heatmap.png", dpi=300)
plt.show()

# Interpretation Helper
for cat in ct.index:
    dominant = ct.loc[cat].idxmax()
    val = ct.loc[cat].max()
    print(f"{cat}: mostly {dominant} ({val:.2f}%)")

# Step 1 — Emotion Breakdown by Severity


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- 0) Paths  ---
PRED_PATH = "/content/test_predictions.csv"     # from your 90% model
VA_PATH   = "/content/Objective1_with_VA_balanced.csv"                # has goem_* + vad_valence/vad_arousal

assert os.path.exists(PRED_PATH), f"Missing predictions at {PRED_PATH}"
assert os.path.exists(VA_PATH),   f"Missing VA/goem file at {VA_PATH}"

# --- 1) Load ---
pred = pd.read_csv(PRED_PATH)
va   = pd.read_csv(VA_PATH)

print("Predictions file columns:", list(pred.columns)[:12], "...")
print("VA+GoEmotions file columns:", list(va.columns)[:12], "...")

# join on text (what used earlier). If that fails, try Post_ID automatically.
join_key = None
if "Text_With_EmojiDesc" in pred.columns and "Text_With_EmojiDesc" in va.columns:
    join_key = "Text_With_EmojiDesc"
elif "Post_ID" in pred.columns and "Post_ID" in va.columns:
    join_key = "Post_ID"
else:
    raise ValueError("Could not find a common join key. Expected Text_With_EmojiDesc or Post_ID in both files.")

# --- 2) Merge predictions with VA/GoEmotions ---
merged = pred.merge(
    va,
    on=join_key,
    how="inner",
    suffixes=("_pred", "_va")
).copy()

# Make sure the predicted severity column exists and is clean
if "pred" not in merged.columns:
    # sometimes saved as 'pred_label' or similar — try to infer
    alt = [c for c in merged.columns if c.lower() in ["pred_label", "pred_severity", "prediction", "predicted_label"]]
    if alt:
        merged["pred"] = merged[alt[0]]
    else:
        raise ValueError("No 'pred' column (predicted severity) found in merged frame.")

merged["pred"] = merged["pred"].astype(str).str.strip()

# --- 3) Detect GoEmotions probability columns ---
goem_cols = [c for c in merged.columns
             if c.startswith("goem_")
             and c not in ("goem_top5_labels", "goem_top5_scores")]

if not goem_cols:
    raise ValueError("No goem_* probability columns found in your VA file. "
                     "Make sure you used the file that still contains them (not a reduced export).")

print(f"Detected {len(goem_cols)} GoEmotions probability columns.")
# Optional: keep only rows that have all goem probabilities present
merged = merged.dropna(subset=goem_cols + ["pred"]).reset_index(drop=True)

# --- 4) Aggregate: mean emotion probabilities per predicted severity ---
severity_order = ["mild_distress", "moderate_distress", "severe_crisis"]
mean_by_sev = (
    merged.groupby("pred")[goem_cols]
    .mean()
    .reindex(severity_order)
)

# Save tidy CSV
out_dir = "/content/emotion_breakdown"
os.makedirs(out_dir, exist_ok=True)
mean_by_sev.to_csv(os.path.join(out_dir, "emotion_means_by_severity.csv"))
print("Saved:", os.path.join(out_dir, "emotion_means_by_severity.csv"))

# --- 5) Top-5 emotions per severity (table) ---
top_rows = []
for sev in severity_order:
    if sev not in mean_by_sev.index:
        continue
    row = mean_by_sev.loc[sev]
    top5 = row.sort_values(ascending=False).head(5)
    for rank, (emo, val) in enumerate(top5.items(), start=1):
        top_rows.append({"pred_severity": sev, "rank": rank, "emotion": emo.replace("goem_", ""), "mean_prob": val})

top5_df = pd.DataFrame(top_rows)
top5_df.to_csv(os.path.join(out_dir, "top5_emotions_by_severity.csv"), index=False)
print("Saved:", os.path.join(out_dir, "top5_emotions_by_severity.csv"))

# --- 6) Quick plots: heatmap + per-severity top10 bars ---
plt.figure(figsize=(12, 6))
# Heatmap can be very wide; show the 12 most discriminative emotions by variance across severities
var_order = mean_by_sev.var(axis=0).sort_values(ascending=False).head(12).index.tolist()
hm = mean_by_sev[var_order]
plt.imshow(hm.values, aspect="auto", cmap="viridis")
plt.colorbar(label="Mean probability")
plt.yticks(range(len(hm.index)), hm.index)
plt.xticks(range(len(hm.columns)), [c.replace("goem_", "") for c in hm.columns], rotation=45, ha="right")
plt.title("GoEmotions (mean probs) by Predicted Severity — Top 12 varying emotions")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "heatmap_goemotions_by_severity.png"), dpi=180)
plt.show()

# Per-severity top10 bar charts
for sev in severity_order:
    if sev not in mean_by_sev.index:
        continue
    top10 = mean_by_sev.loc[sev].sort_values(ascending=False).head(10)
    plt.figure(figsize=(8, 4))
    plt.barh([e.replace("goem_", "") for e in top10.index[::-1]], top10.values[::-1])
    plt.title(f"Top-10 GoEmotions for {sev}")
    plt.xlabel("Mean probability")
    plt.tight_layout()
    fn = os.path.join(out_dir, f"top10_{sev}.png")
    plt.savefig(fn, dpi=180)
    plt.show()
    print("Saved:", fn)

# --- 7)  correlations with V–A for interpretability ---
for target in ["vad_valence", "vad_arousal"]:
    if target in merged.columns:
        # Spearman rank correlations between each emotion prob and valence/arousal
        from scipy.stats import spearmanr
        corrs = []
        for emo in goem_cols:
            rho, p = spearmanr(merged[emo], merged[target])
            corrs.append({"emotion": emo.replace("goem_", ""), "rho": rho, "p": p})
        corr_df = pd.DataFrame(corrs).sort_values("rho", ascending=False)
        corr_df.to_csv(os.path.join(out_dir, f"spearman_{target}.csv"), index=False)
        print("Saved:", os.path.join(out_dir, f"spearman_{target}.csv"))

print("\nDone. Files created in:", out_dir)

# === Step 2: Emotion ↔ Valence/Arousal correlations (with plots) ===
import os, re, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

# --------------------- CONFIG ---------------------
PRED_CSV = "/content/test_predictions.csv"
VA_CSV   = "/content/Objective1_with_VA_balanced.csv"
OUTDIR   = "/content/emotion_va_correlations"
os.makedirs(OUTDIR, exist_ok=True)

# Preferred join key; fall back to Full_Text if needed
JOIN_KEYS = [
    ("pred","Text_With_EmojiDesc","Text_With_EmojiDesc"),
    ("pred","Full_Text","Full_Text"),
]

# Column that holds predicted severity in predictions file
PRED_COL_CANDIDATES = ["pred", "Severity"]  # we prefer 'pred' (your model's label)

# --------------------- LOAD -----------------------
pred = pd.read_csv(PRED_CSV)
va   = pd.read_csv(VA_CSV)

print("Predictions columns:", list(pred.columns)[:12], "...")
print("VA columns:", list(va.columns)[:12], "...")

# Choose the key that exists in both
key = None
for side, l, r in JOIN_KEYS:
    if (l in pred.columns) and (r in va.columns):
        key = (side, r, l)  # (source, va_col, pred_col)
        break
if key is None:
    raise ValueError("Couldn't find a common join key (Text_With_EmojiDesc or Full_Text).")

# Choose severity column from predictions (prefer 'pred')
pred_col = None
for c in PRED_COL_CANDIDATES:
    if c in pred.columns:
        pred_col = c
        break
if pred_col is None:
    raise ValueError("No severity column found in predictions. Expected 'pred' or 'Severity'.")

# Merge
merged = pred.merge(
    va, left_on=key[2], right_on=key[1], how="inner", suffixes=("_pred","_va")
)
print(f"\nJoined on {key[2]} ↔ {key[1]}; merged shape:", merged.shape)

# Make a clean 'pred_severity' column
if pred_col == "Severity" and "pred" in merged.columns:
    merged["pred_severity"] = merged["pred"]
else:
    merged["pred_severity"] = merged[pred_col]

# Ensure V–A columns exist
if not {"vad_valence","vad_arousal"}.issubset(merged.columns):
    raise ValueError("vad_valence/vad_arousal not found in the merged table. "
                     "Please run the V–A mapping step first.")

# Detect GoEmotions probability columns (28)
goem_cols = [c for c in merged.columns
             if c.startswith("goem_")
             and not re.search(r"(top5|labels|scores)", c)]
if len(goem_cols) < 20:
    print("Warning: detected", len(goem_cols), "goem_* columns. "
          "Expected 28; check your input file.")
goem_cols = sorted(goem_cols)

# Drop NAs 
df = merged.dropna(subset=["vad_valence","vad_arousal"] + goem_cols).copy()

# -------------- CORRELATION HELPERS ----------------
def corr_table(frame):
    rows = []
    for emo in goem_cols:
        # Pearson
        r_val  = frame[emo].corr(frame["vad_valence"], method="pearson")
        r_aru  = frame[emo].corr(frame["vad_arousal"], method="pearson")
        # Spearman
        rho_v, pv = spearmanr(frame[emo], frame["vad_valence"])
        rho_a, pa = spearmanr(frame[emo], frame["vad_arousal"])
        rows.append({
            "emotion": emo.replace("goem_",""),
            "pearson_valence": r_val, "pearson_arousal": r_aru,
            "spearman_valence": rho_v, "spearman_val_p": pv,
            "spearman_arousal": rho_a, "spearman_arousal_p": pa
        })
    out = pd.DataFrame(rows).sort_values("pearson_valence", ascending=False)
    return out

def plot_heat(df_corr, title, path_png):
    heat = df_corr.set_index("emotion")[["pearson_valence","pearson_arousal"]]
    plt.figure(figsize=(6.4, max(3.5, 0.22*len(heat))))
    sns.heatmap(heat, annot=False, cmap="coolwarm", center=0, linewidths=.5)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path_png, dpi=200)
    plt.close()

def plot_topbars(df_corr, target, k=8, fname="topbars.png", nice_name="Valence"):
    # target in {"pearson_valence","pearson_arousal"}
    d = df_corr[["emotion", target]].set_index("emotion")
    pos = d[target].sort_values(ascending=False).head(k)
    neg = d[target].sort_values(ascending=True).head(k)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=False)
    pos.sort_values().plot(kind="barh", ax=axes[0], color="#2b8a3e")
    axes[0].set_title(f"Top +{nice_name} correlates")
    axes[0].set_xlabel("Pearson r")

    neg.sort_values().plot(kind="barh", ax=axes[1], color="#c0392b")
    axes[1].set_title(f"Top –{nice_name} correlates")
    axes[1].set_xlabel("Pearson r")

    fig.suptitle(f"GoEmotions ↔ {nice_name} correlations")
    plt.tight_layout()
    p = os.path.join(OUTDIR, fname)
    plt.savefig(p, dpi=200)
    plt.close()
    return p

# ---------------- GLOBAL CORRELATIONS ----------------
corr_all = corr_table(df)
corr_all.to_csv(os.path.join(OUTDIR, "correlations_global.csv"), index=False)

plot_heat(corr_all, "Emotion ↔ Valence/Arousal (Pearson) — ALL",
          os.path.join(OUTDIR,"heat_global.png"))
p1 = plot_topbars(corr_all, "pearson_valence", k=8, fname="topbars_valence.png", nice_name="Valence")
p2 = plot_topbars(corr_all, "pearson_arousal", k=8, fname="topbars_arousal.png", nice_name="Arousal")

print("Saved:",
      os.path.join(OUTDIR,"correlations_global.csv"),
      os.path.join(OUTDIR,"heat_global.png"),
      p1, p2)

# --------------- PER-SEVERITY CORRELATIONS ---------------
persev_paths = []
for sev in ["mild_distress","moderate_distress","severe_crisis"]:
    sub = df[df["pred_severity"] == sev].copy()
    if len(sub) < 50:
        continue
    ct = corr_table(sub)
    outcsv = os.path.join(OUTDIR, f"correlations_{sev}.csv")
    ct.to_csv(outcsv, index=False)
    outpng = os.path.join(OUTDIR, f"heat_{sev}.png")
    plot_heat(ct, f"Emotion ↔ V–A (Pearson) — {sev}", outpng)
    persev_paths += [outcsv, outpng]

print("Per-severity artifacts:", persev_paths)

# --------------- QUICK TEXT SUMMARY (TOP SIGNALS) ---------------
def top_signals(df_corr, k=5):
    vpos = df_corr.nlargest(k, "pearson_valence")[["emotion","pearson_valence"]]
    vneg = df_corr.nsmallest(k, "pearson_valence")[["emotion","pearson_valence"]]
    apos = df_corr.nlargest(k, "pearson_arousal")[["emotion","pearson_arousal"]]
    aneg = df_corr.nsmallest(k, "pearson_arousal")[["emotion","pearson_arousal"]]
    return vpos, vneg, apos, aneg

vpos, vneg, apos, aneg = top_signals(corr_all)
print("\n--- Global signals (Pearson) ---")
print("Valence +:", dict(zip(vpos.emotion, vpos.pearson_valence.round(3))))
print("Valence –:", dict(zip(vneg.emotion, vneg.pearson_valence.round(3))))
print("Arousal +:", dict(zip(apos.emotion, apos.pearson_arousal.round(3))))
print("Arousal –:", dict(zip(aneg.emotion, aneg.pearson_arousal.round(3))))

# Step 3 — Emotion→Severity surrogate model (LogReg + RandomForest)

import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.inspection import permutation_importance

# ----------------------------
# 0) Paths / config
# ----------------------------
CSV_PATH = "/content/Merged_with_VA_and_Predictions.csv"
OUT_DIR  = "/content/surrogate_emotion2severity"
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------
# 1) Load + pick columns
# ----------------------------
df = pd.read_csv(CSV_PATH)

# Find the 28 GoEmotions probability columns
goem_cols = [c for c in df.columns if re.match(r"^goem_[a-z_]+$", c)]
if len(goem_cols) == 0:
    raise ValueError(
        "Could not find columns like 'goem_sadness', 'goem_fear', ... "
        f"in {CSV_PATH}. Please point CSV_PATH to a file that has the 28 GoEmotions columns."
    )

# Decide which severity label to use: prefer model predictions 'pred', else 'Severity' / 'Severity_x'
label_col = None
for cand in ["pred", "Severity", "Severity_x"]:
    if cand in df.columns:
        label_col = cand
        break

if label_col is None:
    raise ValueError("No severity label column found. Expected one of: 'pred', 'Severity', 'Severity_x'.")

print(f"Using label column: {label_col}")
print(f"Found {len(goem_cols)} GoEmotions probability columns.")

# Keep only rows with non-null label
df = df.dropna(subset=[label_col]).reset_index(drop=True)
# Ensure label is string categorical (mild_distress / moderate_distress / severe_crisis)
df[label_col] = df[label_col].astype(str).str.strip()

X = df[goem_cols].astype(float).values
y = df[label_col].values

# Consistent class order for plots
class_order = ["mild_distress", "moderate_distress", "severe_crisis"]
# If some classes missing in this subset, fall back to sorted unique
if not set(class_order).issubset(set(np.unique(y))):
    class_order = sorted(np.unique(y).tolist())
print("Class order:", class_order)

# ----------------------------
# 2) Train/test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ----------------------------
# 3) Logistic Regression (multinomial)
# ----------------------------
logreg = Pipeline([
    ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ("clf", LogisticRegression(
        max_iter=2000, multi_class="multinomial", class_weight="balanced", n_jobs=None
    ))
])
logreg.fit(X_train, y_train)
y_pred_lr = logreg.predict(X_test)

print("\n=== Logistic Regression (GoEmotions→Severity) ===")
print(classification_report(y_test, y_pred_lr, digits=3))
with open(os.path.join(OUT_DIR, "logreg_classification_report.txt"), "w") as f:
    f.write(classification_report(y_test, y_pred_lr, digits=3))

# Coefficients (shape: n_classes × n_features)
coef = logreg.named_steps["clf"].coef_
coef_df = pd.DataFrame(coef, columns=goem_cols, index=logreg.named_steps["clf"].classes_)
coef_df = coef_df.reindex(class_order)  # reorder rows to class_order if present
coef_df.to_csv(os.path.join(OUT_DIR, "logreg_coefficients.csv"))

# Top positive/negative emotions per class (bar charts)
def plot_top_coeffs(coefs_row, title, k=10):
    # coefs_row: pd.Series indexed by goem_cols
    sr = coefs_row.sort_values()
    neg = sr.head(k)
    pos = sr.tail(k)
    fig, ax = plt.subplots(figsize=(8, 6))
    all_ = pd.concat([neg, pos])
    all_.plot(kind="barh", ax=ax, color=["#b23b3b" if v<0 else "#2d7f5e" for v in all_.values])
    ax.axvline(0, color="k", lw=1)
    ax.set_title(title + " — top +/- coefficients (LogReg)")
    ax.set_xlabel("Coefficient (after standardization)")
    fig.tight_layout()
    return fig

for cls in coef_df.index:
    fig = plot_top_coeffs(coef_df.loc[cls], f"{cls}")
    fig.savefig(os.path.join(OUT_DIR, f"logreg_top_coefs_{cls}.png"), dpi=150)
    plt.close(fig)

# ----------------------------
# 4) Random-Forest (feature importance)
# ----------------------------
rf = RandomForestClassifier(
    n_estimators=500, max_depth=None, class_weight="balanced", random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("\n=== Random Forest (GoEmotions→Severity) ===")
print(classification_report(y_test, y_pred_rf, digits=3))
with open(os.path.join(OUT_DIR, "rf_classification_report.txt"), "w") as f:
    f.write(classification_report(y_test, y_pred_rf, digits=3))

# Gini importances
imp = rf.feature_importances_
imp_df = pd.DataFrame({"feature": goem_cols, "rf_importance": imp}).sort_values("rf_importance", ascending=False)
imp_df.to_csv(os.path.join(OUT_DIR, "rf_feature_importances.csv"), index=False)

# Plot top-k RF importances
k = 15
fig, ax = plt.subplots(figsize=(8, 6))
imp_df.head(k).sort_values("rf_importance").plot(
    x="feature", y="rf_importance", kind="barh", ax=ax, legend=False
)
ax.set_title(f"Random-Forest importances (top {k})")
ax.set_xlabel("Gini importance")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "rf_feature_importances_top.png"), dpi=150)
plt.close(fig)

# ----------------------------
# 5)  Permutation importance on the logreg model
# ----------------------------
perm = permutation_importance(
    logreg, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
)
perm_df = pd.DataFrame({
    "feature": goem_cols,
    "perm_importance_mean": perm.importances_mean,
    "perm_importance_std": perm.importances_std,
}).sort_values("perm_importance_mean", ascending=False)
perm_df.to_csv(os.path.join(OUT_DIR, "logreg_permutation_importance.csv"), index=False)

fig, ax = plt.subplots(figsize=(8, 6))
perm_df.head(k).sort_values("perm_importance_mean").plot(
    x="feature", y="perm_importance_mean", kind="barh", ax=ax, legend=False
)
ax.set_title(f"Permutation importance on LogReg (top {k})")
ax.set_xlabel("Mean decrease in score")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "logreg_perm_importances_top.png"), dpi=150)
plt.close(fig)

# ----------------------------
# 6) Summary
# ----------------------------
summary = {
    "dataset_rows": int(df.shape[0]),
    "n_emotion_features": int(len(goem_cols)),
    "label_col_used": label_col,
    "classes": class_order,
    "top_rf_importances": imp_df.head(10).to_dict(orient="records"),
    "top_perm_importances": perm_df.head(10).to_dict(orient="records"),
}
with open(os.path.join(OUT_DIR, "surrogate_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nSaved all outputs to: {OUT_DIR}")
print("Files created:")
for fn in sorted(os.listdir(OUT_DIR)):
    print(" -", fn)
