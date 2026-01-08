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



# ###visualize decision tree

# import matplotlib.pyplot as plt
# from sklearn.tree import plot_tree

# plt.figure(figsize=(50, 20)) # Aumenta l'altezza rispetto alla larghezza per una visualizzazione più verticale
# plot_tree(clf,
#           feature_names=X.columns,
#           class_names=['nonbinder', 'binder'],
#           filled=True,
#           rounded=True,
#           proportion=True,
#           fontsize=20)
# plt.title("Decision Tree Classifier for CDR3 Binding (Vertical Layout)", fontsize=16)
# plt.savefig("decision_tree.png", dpi=300, bbox_inches="tight")
# plt.show()


import re

def cond_to_readable(feature, op, threshold):
    # expects feature like "tcr_pos3_H" or "pep_pos1_+"
    m = re.match(r"(tcr|pep)_pos(\d+)_(H|P|\+|-)$", feature)
    if m and abs(threshold - 0.5) < 1e-6:
        prefix = m.group(1)
        pos = int(m.group(2))
        grp = m.group(3)
        if op == ">":
            return f"({prefix}_pos{pos} == '{grp}')"
        else:  # "<="
            return f"({prefix}_pos{pos} != '{grp}')"
    # fallback
    return f"({feature} {op} {threshold:g})"

def rules_to_dnf(rules):
    clauses = []
    for r in rules:
        clause = " and ".join(cond_to_readable(f,op,t) for f,op,t in r["conditions"])
        clauses.append(f"({clause})")
    return " or ".join(clauses) if clauses else "False"

import numpy as np

def extract_tree_rules(clf, feature_names, class_names=None, target_class=1):
    """
    Extract root->leaf rules from a fitted sklearn DecisionTreeClassifier.

    Returns a list of dicts:
      {
        "class": predicted_class,
        "class_name": ...,
        "conditions": [(feature, op, threshold), ...],
        "n_samples": int,
        "value": class_counts (array),
        "proba": class_probabilities (array)
      }
    """
    tree = clf.tree_
    feat = tree.feature
    thr = tree.threshold
    children_left = tree.children_left
    children_right = tree.children_right
    value = tree.value  # shape: (n_nodes, 1, n_classes)
    n_node_samples = tree.n_node_samples

    rules = []

    def recurse(node, path):
        is_leaf = (children_left[node] == children_right[node])
        if is_leaf:
            counts = value[node][0]
            pred_class = int(np.argmax(counts))
            if pred_class == target_class:
                probs = counts / counts.sum() if counts.sum() > 0 else counts
                rules.append({
                    "class": pred_class,
                    "class_name": class_names[pred_class] if class_names else str(pred_class),
                    "conditions": path.copy(),
                    "n_samples": int(n_node_samples[node]),
                    "value": counts.copy(),
                    "proba": probs.copy(),
                })
            return

        f_idx = feat[node]
        f_name = feature_names[f_idx]
        t = thr[node]

        # left: feature <= threshold
        recurse(children_left[node], path + [(f_name, "<=", float(t))])

        # right: feature > threshold
        recurse(children_right[node], path + [(f_name, ">", float(t))])

    recurse(0, [])
    return rules

feature_names = list(X.columns)
rules = extract_tree_rules(clf, feature_names, class_names=["nonbinder","binder"], target_class=1)

for i, r in enumerate(rules, 1):
    print(f"\nRule {i} (n={r['n_samples']}, proba={r['proba']}):")
    for f, op, t in r["conditions"]:
        print("  ", cond_to_readable(f, op, t))

print("\nDNF:")
print(rules_to_dnf(rules)[0:100])


def peptide_fixed_assignment(peptide, all_feature_names, plus, minus, hydro, polar):
    def aa2grp(a):
        if a in plus:  return "+"
        if a in minus: return "-"
        if a in hydro: return "H"
        return "P"

    # Start with all pep_* features set to 0
    fixed = {fn: 0 for fn in all_feature_names if fn.startswith("pep_")}

    # Set the 1s for positions that exist in this peptide
    for i, a in enumerate(peptide):
        g = aa2grp(a)
        for cat in ["H", "P", "+", "-"]:
            name = f"pep_pos{i}_{cat}"
            if name in fixed:
                fixed[name] = int(g == cat)

    return fixed


import numpy as np

def conditioned_tree_to_dnf_over_tcr(clf, feature_names, fixed_pep, positive_class=1):
    """
    Returns DNF clauses over NON-fixed features (i.e. tcr_*),
    by following peptide splits deterministically.
    Each clause: dict {feature_name: 0/1}
    Whole formula: OR over clauses.
    """
    tree = clf.tree_
    feat = tree.feature
    thr = tree.threshold
    left = tree.children_left
    right = tree.children_right
    value = tree.value

    clauses = []

    def is_leaf(n):
        return left[n] == right[n] == -1

    def leaf_class(n):
        return int(np.argmax(value[n][0]))

    def go(n, constraints):
        if is_leaf(n):
            if leaf_class(n) == positive_class:
                clause = {}
                ok = True
                for name, val in constraints:
                    if name in clause and clause[name] != val:
                        ok = False
                        break
                    clause[name] = val
                if ok:
                    clauses.append(clause)
            return

        fidx = feat[n]
        name = feature_names[fidx]
        t = thr[n]

        # (Your features are 0/1, so thresholds are basically 0.5)
        if name in fixed_pep:
            v = fixed_pep[name]
            # left: x <= t  (v==0 for t=0.5), right: x > t (v==1)
            if v <= t:
                go(left[n], constraints)
            else:
                go(right[n], constraints)
        else:
            # keep this split as a boolean literal in the clause
            go(left[n],  constraints + [(name, 0)])
            go(right[n], constraints + [(name, 1)])

    go(0, [])
    # Optionally: filter to only tcr_ literals (should already be, if fixed_pep covers all pep_)
    clauses = [{k:v for k,v in c.items() if k.startswith("tcr_")} for c in clauses]
    return clauses

def dnf_to_string(clauses, op_and=" & ", op_or=" | ", neg="~"):
    if not clauses:
        return "FALSE"
    parts = []
    for clause in clauses:
        lits = [(k if v == 1 else f"{neg}{k}") for k, v in sorted(clause.items())]
        parts.append("(" + op_and.join(lits) + ")")
    return op_or.join(parts)

# choose an unseen peptide (string) you want rules for
unseen_pep = "NLVPMVATV" #"CASSALASGGDTQYF" # example; use your peptide of interest

fixed_pep = peptide_fixed_assignment(
    unseen_pep,
    all_feature_names=list(X.columns),
    plus=plus, minus=minus, hydro=hydro, polar=polar
)

clauses_tcr = conditioned_tree_to_dnf_over_tcr(clf, list(X.columns), fixed_pep, positive_class=1)
formula_tcr_only = dnf_to_string(clauses_tcr)

print("Number of positive clauses (for this peptide):", len(clauses_tcr))
print(formula_tcr_only[0:100])



