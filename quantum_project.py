import math
import re
from typing import List, Tuple

import numpy as np
import pandas as pd
import scirpy as ir
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import MCXGate
from qiskit_aer import Aer
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

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


###quantum

# -------------------------
# 2-bit class encoding per position
# -------------------------
# 00 -> H (hydrophobic)
# 01 -> P (polar)
# 10 -> + (positively charged)
# 11 -> - (negatively charged)
BITS_TO_CLASS = {"00": "H", "01": "P", "10": "+", "11": "-"}
CLASS_TO_BITS = {v: k for k, v in BITS_TO_CLASS.items()}

# Data qubits (10): 5 positions × 2 bits
# pos0: q0=a0, q1=b0
# pos1: q2=a1, q3=b1
# pos2: q4=a2, q5=b2
# pos3: q6=a3, q7=b3
# pos4: q8=a4, q9=b4
#
# Ancillas:
# q10 = check ancilla #1 (parity check)
# q11 = check ancilla #2 (parity check)
#
# NOTE: We no longer use a phase ancilla |->. The oracle is implemented as an MCZ
# on the data qubits (via H–MCX–H), freeing q11 for checking.

CH_POS = set("KR")
CH_NEG = set("DE")

def peptide_core(peptide: str, k: int = 3) -> str:
    p = peptide.strip().upper()
    if len(p) <= k:
        return p
    start = (len(p) - k) // 2
    return p[start:start + k]

def peptide_charge_sign(core: str) -> str:
    pos = sum(aa in CH_POS for aa in core)
    neg = sum(aa in CH_NEG for aa in core)
    if neg > pos:
        return "neg"
    if pos > neg:
        return "pos"
    return "none"

def decode_10bit_to_classes(data10_q0_to_q9: str) -> List[str]:
    return [BITS_TO_CLASS[data10_q0_to_q9[i:i+2]] for i in range(0, 10, 2)]

# -------------------------
# Predicate (classical)
# -------------------------
def predicate_classical(data10: str, peptide: str) -> bool:
    """
    More biologically interpretable toy rules:

    R1: pos0 == H
    R2: pos4 == H
    R3: pos2 == P
    R4: peptide-conditioned electrostatic contact:
        - if peptide core net neg -> pos1 == +
        - if peptide core net pos -> pos1 == -
        - if neutral -> pos1 is charged (+ or -)
    R5: pos3 not charged (avoid overly charged core): a3 == 0
    """
    if len(data10) != 10 or any(ch not in "01" for ch in data10):
        return False

    # unpack bits
    a0, b0 = int(data10[0]), int(data10[1])
    a1, b1 = int(data10[2]), int(data10[3])
    a2, b2 = int(data10[4]), int(data10[5])
    a3, b3 = int(data10[6]), int(data10[7])
    a4, b4 = int(data10[8]), int(data10[9])

    # R1/R2: anchors
    pos0_is_H = (a0 == 0 and b0 == 0)
    pos4_is_H = (a4 == 0 and b4 == 0)

    # R3: center polar (01)
    pos2_is_P = (a2 == 0 and b2 == 1)

    # R5: pos3 not charged -> a3 == 0
    pos3_not_charged = (a3 == 0)

    # R4: peptide-conditioned pos1 charge/sign
    core = peptide_core(peptide, 3)
    sign = peptide_charge_sign(core)

    if sign == "neg":
        # pos1 == +  (10)
        pos1_ok = (a1 == 1 and b1 == 0)
    elif sign == "pos":
        # pos1 == -  (11)
        pos1_ok = (a1 == 1 and b1 == 1)
    else:
        # neutral -> pos1 charged (a1 == 1), b1 can be 0/1
        pos1_ok = (a1 == 1)

    return pos0_is_H and pos4_is_H and pos2_is_P and pos1_ok and pos3_not_charged


def brute_force_solutions(peptide: str, limit_print: int = 12) -> Tuple[int, List[str]]:
    sols = []
    for x in range(2**10):
        s = format(x, "010b")  # q0..q9 order
        if predicate_classical(s, peptide):
            sols.append(s)

    print(f"[bruteforce] Peptide={peptide} core={peptide_core(peptide,3)} sign={peptide_charge_sign(peptide_core(peptide,3))}")
    print(f"[bruteforce] M = {len(sols)} solutions out of N = 1024")
    for s in sols[:limit_print]:
        print(" ", s, "->", decode_10bit_to_classes(s))
    if len(sols) > limit_print:
        print(f"... (showing first {limit_print} of {len(sols)})")
    return len(sols), sols

# -------------------------
# Quantum phase oracle for the SAME predicate (NO phase ancilla)
# -------------------------
def phase_oracle_predicate(peptide: str) -> QuantumCircuit:
    """
    Implements a phase flip on the data register iff predicate holds,
    using an MCZ realized as H(target) - MCX(controls->target) - H(target).

    Uses ONLY data qubits; q10/q11 are left untouched for checks.
    """
    qc = QuantumCircuit(12, name="Oracle(bio,MCZ)")

    # Indices (data)
    a0, b0 = 0, 1
    a1, b1 = 2, 3
    a2, b2 = 4, 5
    a3, b3 = 6, 7
    a4, b4 = 8, 9

    # Determine which pos1 condition applies
    core = peptide_core(peptide, 3)
    sign = peptide_charge_sign(core)

    # Apply X for 0-controls (map desired 0-literals into 1-controls)
    qc.x(a0); qc.x(b0)  # pos0 == 00
    qc.x(a4); qc.x(b4)  # pos4 == 00
    qc.x(a2)            # pos2 == 01 => a2=0, b2=1
    qc.x(a3)            # pos3 not charged => a3=0

    # Base controls (must be |1> after mapping)
    controls = [a0, b0, a4, b4, a2, b2, a3]  # b2 is required 1, no X on b2

    # pos1 controls depending on peptide sign
    if sign == "neg":
        # pos1 == 10 => a1=1, b1=0
        qc.x(b1)         # map b1=0 to b1=1 control
        controls += [a1, b1]
    elif sign == "pos":
        # pos1 == 11 => a1=1, b1=1
        controls += [a1, b1]
    else:
        # neutral => a1=1 (charged)
        controls += [a1]

    # Multi-controlled Z on all controls:
    # pick a target qubit from the controls list; apply H-target, MCX, H-target.
    # This phase-flips exactly when ALL controls are 1.
    if len(controls) < 2:
        # (Shouldn't happen here, but keep it safe.)
        qc.z(controls[0])
    else:
        target = controls[-1]
        ctrl_rest = controls[:-1]
        qc.h(target)
        qc.append(MCXGate(len(ctrl_rest)), ctrl_rest + [target])
        qc.h(target)

    # Uncompute X-maps
    if sign == "neg":
        qc.x(b1)

    qc.x(a3)
    qc.x(a2)
    qc.x(b4); qc.x(a4)
    qc.x(b0); qc.x(a0)

    return qc

# -------------------------
# Grover diffuser on data register only (q0..q9)
# -------------------------
def diffuser_on_data_10() -> QuantumCircuit:
    qc = QuantumCircuit(12, name="Diffuser(data10)")
    data = list(range(10))

    qc.h(data)
    qc.x(data)

    target = data[-1]
    controls = data[:-1]
    qc.h(target)
    qc.append(MCXGate(len(controls)), controls + [target])
    qc.h(target)

    qc.x(data)
    qc.h(data)
    return qc

# -------------------------
# Add lightweight error-detection checks (2 parities) using q10,q11
# -------------------------
def add_anchor_parity_checks(qc: QuantumCircuit, check1: int = 10, check2: int = 11):
    """
    Compute two simple parity checks that SHOULD be 0 for valid states:
      check1 = a0 XOR a4  (q0 XOR q8)
      check2 = b0 XOR b4  (q1 XOR q9)

    Measure these ancillas and postselect on 00.
    """
    qc.reset(check1)
    qc.reset(check2)

    # check1 = a0 XOR a4
    qc.cx(0, check1)
    qc.cx(8, check1)

    # check2 = b0 XOR b4
    qc.cx(1, check2)
    qc.cx(9, check2)

# -------------------------
# Build Grover circuit
# -------------------------
def build_grover_circuit(peptide: str, iters: int) -> QuantumCircuit:
    oracle = phase_oracle_predicate(peptide)
    diff = diffuser_on_data_10()

    # Measure: 10 data bits + 2 check bits
    qc = QuantumCircuit(12, 12)
    data = list(range(10))
    check1, check2 = 10, 11

    # data superposition
    qc.h(data)

    for _ in range(iters):
        qc.append(oracle, range(12))
        qc.append(diff, range(12))

    # Compute check bits at the end (error detection via postselection)
    add_anchor_parity_checks(qc, check1=check1, check2=check2)

    # Measure q0..q9 -> c0..c9
    qc.measure(data, list(range(10)))
    # Measure checks -> c10,c11
    qc.measure(check1, 10)
    qc.measure(check2, 11)

    assert qc.num_qubits == 12
    return qc


# -------------------------
# Run (simulator) + compare to brute force
# -------------------------
if __name__ == "__main__":
    peptide = "GLCTLVAML"  # replace

    M, sols = brute_force_solutions(peptide, limit_print=8)
    N = 2**10

    # Good iteration heuristic
    iters = max(1, int(round((math.pi / 4) * math.sqrt(N / max(1, M)))))
    print(f"[grover] Using iters = {iters} (based on M={M})")

    qc = build_grover_circuit(peptide, iters=iters)
    print("[grover] Circuit qubits:", qc.num_qubits)

    backend = Aer.get_backend("aer_simulator")
    tqc = transpile(qc, backend=backend, optimization_level=1)
    res = backend.run(tqc, shots=2048).result()
    counts = res.get_counts()

    def key_to_c0_to_c11(key: str) -> str:
        # Qiskit returns c11..c0 as a string; reverse to get c0..c11
        return key[::-1]

    def split_data_and_checks(key: str) -> Tuple[str, str]:
        c = key_to_c0_to_c11(key)
        data10 = c[:10]       # c0..c9 corresponds to q0..q9 measurements
        checks = c[10:12]     # c10..c11 corresponds to q10,q11 checks
        return data10, checks

    top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:15]
    print("\nTop measured results (showing check bits):")
    for key, ct in top:
        data10, checks = split_data_and_checks(key)
        ok = predicate_classical(data10, peptide)
        chk_ok = (checks == "00")
        tag = "VALID+CHK" if (ok and chk_ok) else ("VALID" if ok else ("CHK" if chk_ok else ""))
        print(f"{key}  {ct:4d}  q0..q9={data10}  checks(c10,c11)={checks}  {decode_10bit_to_classes(data10)}  {tag}")

    # Raw VALID rate (ignoring checks)
    valid_shots = 0
    for key, ct in counts.items():
        data10, _ = split_data_and_checks(key)
        if predicate_classical(data10, peptide):
            valid_shots += ct
    print(f"\nVALID shots (raw) = {valid_shots} / 2048  ({valid_shots/2048:.3f})")

    # Postselected VALID rate (keep only checks==00)
    kept_shots = 0
    valid_kept_shots = 0
    for key, ct in counts.items():
        data10, checks = split_data_and_checks(key)
        if checks == "00":
            kept_shots += ct
            if predicate_classical(data10, peptide):
                valid_kept_shots += ct

    if kept_shots > 0:
        print(f"Postselected on checks==00: kept = {kept_shots} / 2048  ({kept_shots/2048:.3f})")
        print(f"VALID among kept = {valid_kept_shots} / {kept_shots}  ({valid_kept_shots/kept_shots:.3f})")
    else:
        print("Postselected on checks==00: kept = 0 (no shots passed checks)")
