import pandas as pd
import numpy as np
import scirpy as ir

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import roc_auc_score, accuracy_score

### Create decision tree

# ------------------------------------------------------------
# 1) Load VDJdb
# ------------------------------------------------------------
adata = ir.datasets.vdjdb()
df = adata.obs.copy()

cdr3b = ir.get.airr(adata, "junction_aa", chain="VDJ_1")
locus = ir.get.airr(adata, "locus", chain="VDJ_1")

df = df.join(cdr3b.rename("cdr3b")).join(locus.rename("locus"))
df = df.dropna(subset=["cdr3b", "locus"])
df = df[df["locus"].str.upper() == "TRB"]

# ------------------------------------------------------------
# 2) Detect epitope + MHC columns
# ------------------------------------------------------------
def find_col(columns, tokens):
    for c in columns:
        low = c.lower()
        if all(t in low for t in tokens):
            return c
    return None

epitope_col = find_col(df.columns, ["epitope"])
mhc_col = find_col(df.columns, ["mhc"]) or find_col(df.columns, ["hla"])

if epitope_col is None or mhc_col is None:
    raise ValueError("Could not detect epitope or MHC column")

# ------------------------------------------------------------
# 3) Restrict to one MHC allele (important!)
# ------------------------------------------------------------
df = df[df[mhc_col].astype(str).str.contains("HLA-A\\*02:01", na=False)]
df = df.dropna(subset=[epitope_col])

# Deduplicate TCR–epitope pairs
df = df.drop_duplicates(["cdr3b", epitope_col])

# ------------------------------------------------------------
# 4) Define feature encoders
# ------------------------------------------------------------
plus  = set("KRH")
minus = set("DE")
hydro = set("AVILMFWY")
polar = set("STNQGPC")

def aa2grp(a):
    if a in plus:  return "+"
    if a in minus: return "-"
    if a in hydro: return "H"
    return "P"

def center5(seq):
    if len(seq) < 5:
        return None
    mid = len(seq) // 2
    return seq[mid-2:mid+3]

def featurize_window(w, prefix):
    feats = {}
    for i, a in enumerate(w):
        g = aa2grp(a)
        for cat in ["H", "P", "+", "-"]:
            feats[f"{prefix}_pos{i}_{cat}"] = int(g == cat)
    return feats

# ------------------------------------------------------------
# 5) Build positive and mismatch-negative pairs
# ------------------------------------------------------------
rows = []

epitopes = df[epitope_col].unique()

for _, r in df.iterrows():
    w5 = center5(r["cdr3b"])
    if w5 is None:
        continue

    # positive
    rows.append({
        "cdr3b": r["cdr3b"],
        "epitope": r[epitope_col],
        "label": 1,
        "tcr_feats": featurize_window(w5, "tcr"),
        "pep_feats": featurize_window(r[epitope_col], "pep")
    })

    # negatives: mismatched epitopes
    neg_eps = np.random.choice(
        epitopes[epitopes != r[epitope_col]],
        size=min(3, len(epitopes)-1),
        replace=False
    )

    for e in neg_eps:
        rows.append({
            "cdr3b": r["cdr3b"],
            "epitope": e,
            "label": 0,
            "tcr_feats": featurize_window(w5, "tcr"),
            "pep_feats": featurize_window(e, "pep")
        })

# ------------------------------------------------------------
# 6) Build feature matrix
# ------------------------------------------------------------
X = []
y = []
epitope_ids = []

for r in rows:
    feats = {}
    feats.update(r["tcr_feats"])
    feats.update(r["pep_feats"])
    X.append(feats)
    y.append(r["label"])
    epitope_ids.append(r["epitope"])

X = pd.DataFrame(X).fillna(0).astype(int)
y = np.array(y)
epitope_ids = np.array(epitope_ids)

# ------------------------------------------------------------
# 7) Epitope-held-out split (CRITICAL)
# ------------------------------------------------------------
unique_eps = np.unique(epitope_ids)

train_eps, test_eps = train_test_split(
    unique_eps, test_size=0.2, random_state=0
)

train_mask = np.isin(epitope_ids, train_eps)
test_mask  = np.isin(epitope_ids, test_eps)

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

# ------------------------------------------------------------
# 8) Train interpretable model
# ------------------------------------------------------------
clf = DecisionTreeClassifier(
    max_depth=20,
    min_samples_leaf=2,
    random_state=0
)

clf.fit(X_train, y_train)

# ------------------------------------------------------------
# 9) Evaluate on unseen epitopes
# ------------------------------------------------------------
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:,1]

print("Unseen-epitope accuracy:", accuracy_score(y_test, y_pred))
print("Unseen-epitope AUC:     ", roc_auc_score(y_test, y_prob))

print("\nLearned rules:\n")
print(export_text(clf, feature_names=list(X.columns)))



###visualize decision tree

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(50, 20)) # Aumenta l'altezza rispetto alla larghezza per una visualizzazione più verticale
plot_tree(clf,
          feature_names=X.columns,
          class_names=['nonbinder', 'binder'],
          filled=True,
          rounded=True,
          proportion=True,
          fontsize=20)
plt.title("Decision Tree Classifier for CDR3 Binding (Vertical Layout)", fontsize=16)
plt.show()