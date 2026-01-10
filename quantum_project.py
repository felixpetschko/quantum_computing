"""End-to-end pipeline: train interpretable TCR/epitope rules and run Grover search."""

import argparse
import math
import os
import random
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
from sklearn.tree import DecisionTreeClassifier

SEED = 0
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

VISUALIZE_TREE = False
SHOW_CONFUSION_MATRIX = False
SHOW_TREE_TEXT = False
MAX_RULES_TO_PRINT = 20
DEFAULT_PEPTIDE = "DVWQKSLTM"

### Helpers and configuration

# --- Column detection helper ---
def find_col(columns, tokens):
    """Return the first column name containing all tokens (case-insensitive)."""
    for c in columns:
        low = c.lower()
        if all(t in low for t in tokens):
            return c
    return None

# --- Feature encoding helpers ---
plus  = set("KRH")
minus = set("DE")
hydro = set("AVILMFWY")
polar = set("STNQGPC")

def aa2grp(a):
    """Map an amino acid to a coarse physicochemical group."""
    if a in plus:  return "+"
    if a in minus: return "-"
    if a in hydro: return "H"
    return "P"

def center5(seq):
    """Return the central 5-mer of a sequence, or None if too short."""
    if len(seq) < 5:
        return None
    mid = len(seq) // 2
    return seq[mid-2:mid+3]

def featurize_window(w, prefix):
    """One-hot encode a window of amino-acid groups with a prefix."""
    feats = {}
    for i, a in enumerate(w):
        g = aa2grp(a)
        for cat in ["H", "P", "+", "-"]:
            feats[f"{prefix}_pos{i}_{cat}"] = int(g == cat)
    return feats

def cond_to_readable(feature, op, threshold):
    """Format a tree split into a readable condition string."""
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
    """Convert a list of rule dicts into a DNF string for display."""
    clauses = []
    for r in rules:
        clause = " and ".join(cond_to_readable(f,op,t) for f,op,t in r["conditions"])
        clauses.append(f"({clause})")
    return " or ".join(clauses) if clauses else "False"

def extract_tree_rules(clf, feature_names, class_names=None, target_class=1):
    """Extract root-to-leaf rules for a target class from a decision tree."""
    tree = clf.tree_
    feat = tree.feature
    thr = tree.threshold
    children_left = tree.children_left
    children_right = tree.children_right
    value = tree.value  # shape: (n_nodes, 1, n_classes)
    n_node_samples = tree.n_node_samples

    rules = []

    def recurse(node, path):
        """Depth-first traversal to collect rules from root to leaves."""
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

def peptide_fixed_assignment(peptide, all_feature_names, plus, minus, hydro, polar):
    """Build a fixed peptide feature assignment for conditioning the tree."""
    def aa2grp(a):
        """Local aa-to-group mapping for peptide encoding."""
        if a in plus:  return "+"
        if a in minus: return "-"
        if a in hydro: return "H"
        return "P"

    # Start with all pep_* features set to 0.
    fixed = {fn: 0 for fn in all_feature_names if fn.startswith("pep_")}

    # Set the 1s for positions that exist in this peptide.
    for i, a in enumerate(peptide):
        g = aa2grp(a)
        for cat in ["H", "P", "+", "-"]:
            name = f"pep_pos{i}_{cat}"
            if name in fixed:
                fixed[name] = int(g == cat)

    return fixed


def conditioned_tree_to_dnf_over_tcr(clf, feature_names, fixed_pep, positive_class=1):
    """Return DNF clauses over tcr_* features by conditioning on a peptide."""
    tree = clf.tree_
    feat = tree.feature
    thr = tree.threshold
    left = tree.children_left
    right = tree.children_right
    value = tree.value

    clauses = []

    def is_leaf(n):
        """Return True if the node is a leaf."""
        return left[n] == right[n] == -1

    def leaf_class(n):
        """Return the predicted class index at a leaf node."""
        return int(np.argmax(value[n][0]))

    def go(n, constraints):
        """Recurse through the tree while collecting clause literals."""
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

        # Features are 0/1, so thresholds are effectively 0.5.
        if name in fixed_pep:
            v = fixed_pep[name]
            # left: x <= t (v==0 for t=0.5), right: x > t (v==1)
            if v <= t:
                go(left[n], constraints)
            else:
                go(right[n], constraints)
        else:
            # Keep this split as a boolean literal in the clause.
            go(left[n],  constraints + [(name, 0)])
            go(right[n], constraints + [(name, 1)])

    go(0, [])
    # Filter to only tcr_ literals (pep_ literals are fixed by conditioning).
    clauses = [{k:v for k,v in c.items() if k.startswith("tcr_")} for c in clauses]
    return clauses

def dnf_to_string(clauses, op_and=" & ", op_or=" | ", neg="~"):
    """Format clause dicts as a DNF string with customizable operators."""
    if not clauses:
        return "FALSE"
    parts = []
    for clause in clauses:
        lits = [(k if v == 1 else f"{neg}{k}") for k, v in sorted(clause.items())]
        parts.append("(" + op_and.join(lits) + ")")
    return op_or.join(parts)

def clauses_for_peptide(peptide):
    """Generate TCR-only clauses for a given peptide with handcrafted filters."""
    peptide_core = center5(peptide)
    if peptide_core is None:
        raise ValueError("Peptide must be at least 5 amino acids long.")

    fixed_pep = peptide_fixed_assignment(
        peptide_core,
        all_feature_names=list(X.columns),
        plus=plus, minus=minus, hydro=hydro, polar=polar
    )
    clauses = conditioned_tree_to_dnf_over_tcr(clf, list(X.columns), fixed_pep, positive_class=1)
    clauses = apply_same_anchor_rule(clauses)

    return clauses

def apply_same_anchor_rule(clauses):
    """Expand clauses to enforce pos0 == pos4 across H/P/+/- classes."""
    same_anchor = []
    for c in clauses:
        for cat in ["H", "P", "+", "-"]:
            same_anchor.append(dict(c, **{
                f"tcr_pos0_{cat}": 1,
                f"tcr_pos4_{cat}": 1,
            }))
    return same_anchor

### Quantum utilities

# --- 2-bit class encoding per position ---
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
# NOTE: The oracle is an MCZ on data qubits (via H–MCX–H); no phase ancilla.

def decode_10bit_to_classes(data10_q0_to_q9: str) -> List[str]:
    """Decode a 10-bit string into five class symbols (H/P/+/-)."""
    return [BITS_TO_CLASS[data10_q0_to_q9[i:i+2]] for i in range(0, 10, 2)]

def clause_accepts_data10(clause, data10: str) -> bool:
    """Return True if a 10-bit assignment satisfies a clause dict."""
    classes = decode_10bit_to_classes(data10)
    feats = {}
    for pos, cls in enumerate(classes):
        for cat in ["H", "P", "+", "-"]:
            feats[f"tcr_pos{pos}_{cat}"] = int(cls == cat)
    return all(feats.get(k) == v for k, v in clause.items())

def predicate_from_clauses(data10: str, clauses) -> bool:
    """Return True if any clause matches the 10-bit assignment."""
    return any(clause_accepts_data10(c, data10) for c in clauses)

def clause_to_controls(clause):
    """
    Convert a clause dict into exact bit controls for q0..q9.

    Returns a list of (qubit_index, bit_value) controls or None if
    the clause has conflicting positive literals.
    """
    pos_to_class = {}
    for feat, val in clause.items():
        m = re.match(r"tcr_pos(\d+)_(H|P|\+|-)$", feat)
        if not m:
            continue
        pos = int(m.group(1))
        cat = m.group(2)
        if val == 1:
            if pos in pos_to_class and pos_to_class[pos] != cat:
                return None
            pos_to_class[pos] = cat

    controls = []
    for pos, cat in sorted(pos_to_class.items()):
        bits = CLASS_TO_BITS[cat]
        q0 = pos * 2
        q1 = q0 + 1
        controls.append((q0, int(bits[0])))
        controls.append((q1, int(bits[1])))
    return controls

def expand_clauses_for_oracle(clauses, positions=range(5)):
    """
    Expand negative literals into OR over remaining classes.

    Returns clauses with explicit positive class assignments for each position.
    """
    expanded = []
    all_cats = ["H", "P", "+", "-"]

    for clause in clauses:
        pos_pos = {p: set() for p in positions}
        pos_neg = {p: set() for p in positions}
        other = {}

        for feat, val in clause.items():
            m = re.match(r"tcr_pos(\d+)_(H|P|\+|-)$", feat)
            if not m:
                other[feat] = val
                continue
            pos = int(m.group(1))
            cat = m.group(2)
            if pos not in pos_pos:
                continue
            if val == 1:
                pos_pos[pos].add(cat)
            else:
                pos_neg[pos].add(cat)

        base = dict(other)
        multi_positions = []

        for pos in positions:
            allowed = set(all_cats) - pos_neg[pos]
            if pos_pos[pos]:
                allowed = allowed.intersection(pos_pos[pos])
            if not allowed:
                base = None
                break
            if pos_pos[pos]:
                # Conflicting positives for the same position make the clause unsatisfiable.
                if len(pos_pos[pos]) > 1:
                    base = None
                    break
                # already fixed by a positive literal
                cat = next(iter(allowed))
                base[f"tcr_pos{pos}_{cat}"] = 1
            elif pos_neg[pos]:
                # expand negatives into explicit positives
                if len(allowed) == 1:
                    cat = next(iter(allowed))
                    base[f"tcr_pos{pos}_{cat}"] = 1
                else:
                    multi_positions.append((pos, sorted(allowed)))

        if base is None:
            continue

        if not multi_positions:
            expanded.append(base)
            continue

        clauses_acc = [base]
        for pos, allowed in multi_positions:
            next_acc = []
            for c in clauses_acc:
                for cat in allowed:
                    nc = dict(c)
                    nc[f"tcr_pos{pos}_{cat}"] = 1
                    next_acc.append(nc)
            clauses_acc = next_acc
        expanded.extend(clauses_acc)

    return expanded

def phase_oracle_from_clauses(clauses) -> QuantumCircuit:
    """
    Phase-flip states that satisfy ANY clause.
    Clauses are expected to have explicit positive assignments per position.
    """
    qc = QuantumCircuit(12, name="Oracle(clauses)")
    for clause in clauses:
        controls = clause_to_controls(clause)
        if not controls:
            continue
        toggled = []
        qubits = []
        for q, bit in controls:
            if bit == 0:
                qc.x(q)
                toggled.append(q)
            qubits.append(q)

        if len(qubits) == 1:
            qc.z(qubits[0])
        else:
            target = qubits[-1]
            ctrl_rest = qubits[:-1]
            qc.h(target)
            qc.append(MCXGate(len(ctrl_rest)), ctrl_rest + [target])
            qc.h(target)

        for q in reversed(toggled):
            qc.x(q)
    return qc

# --- Predicate helpers ---
def brute_force_solutions(clauses, limit_print: int = 12) -> Tuple[int, List[str]]:
    """Exhaustively count satisfying 10-bit assignments for sanity checks."""
    sols = []
    for x in range(2**10):
        s = format(x, "010b")  # q0..q9 order
        if predicate_from_clauses(s, clauses):
            sols.append(s)

    print(f"[bruteforce] M = {len(sols)} solutions out of N = 1024")
    for s in sols[:limit_print]:
        print(" ", s, "->", decode_10bit_to_classes(s))
    if len(sols) > limit_print:
        print(f"... (showing first {limit_print} of {len(sols)})")
    return len(sols), sols

# --- Grover components ---
def diffuser_on_data_10() -> QuantumCircuit:
    """Return a Grover diffuser acting only on q0..q9."""
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

# --- Error-detection checks (ancillas) ---
def add_anchor_parity_checks(qc: QuantumCircuit, check1: int = 10, check2: int = 11):
    """
    Compute two simple parity checks that should be 0 for valid states.

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

# --- Grover circuit builder ---
def build_grover_circuit_from_clauses(clauses, iters: int) -> QuantumCircuit:
    """Construct a Grover circuit with a clause-based phase oracle."""
    oracle = phase_oracle_from_clauses(clauses)
    diff = diffuser_on_data_10()

    qc = QuantumCircuit(12, 12)
    data = list(range(10))
    check1, check2 = 10, 11

    qc.h(data)
    for _ in range(iters):
        qc.append(oracle, range(12))
        qc.append(diff, range(12))

    add_anchor_parity_checks(qc, check1=check1, check2=check2)
    qc.measure(data, list(range(10)))
    qc.measure(check1, 10)
    qc.measure(check2, 11)
    return qc


def main(peptide=None):
    """Run the full training, rule extraction, and quantum search pipeline."""
    global X, y, epitope_ids, clf

    #Load VDJdb.
    adata = ir.datasets.vdjdb()
    df = adata.obs.copy()

    cdr3b = ir.get.airr(adata, "junction_aa", chain="VDJ_1")
    locus = ir.get.airr(adata, "locus", chain="VDJ_1")

    df = df.join(cdr3b.rename("cdr3b")).join(locus.rename("locus"))
    df = df.dropna(subset=["cdr3b", "locus"])
    df = df[df["locus"].str.upper() == "TRB"]

    #Detect epitope + MHC columns.
    epitope_col = find_col(df.columns, ["epitope"])
    mhc_col = find_col(df.columns, ["mhc"]) or find_col(df.columns, ["hla"])

    if epitope_col is None or mhc_col is None:
        raise ValueError("Could not detect epitope or MHC column")

    #Restrict to one MHC allele.
    df = df[df[mhc_col].astype(str).str.contains("HLA-A\\*02:01", na=False)]
    df = df.dropna(subset=[epitope_col])

    # Deduplicate TCR-epitope pairs.
    df = df.drop_duplicates(["cdr3b", epitope_col])

    #Build positive and mismatch-negative pairs.
    rows = []

    epitopes = df[epitope_col].unique()

    for _, r in df.iterrows():
        w5 = center5(r["cdr3b"])
        if w5 is None:
            continue
        pep_w5 = center5(r[epitope_col])
        if pep_w5 is None:
            continue

        # Positive example.
        rows.append({
            "cdr3b": r["cdr3b"],
            "epitope": r[epitope_col],
            "label": 1,
            "tcr_feats": featurize_window(w5, "tcr"),
            "pep_feats": featurize_window(pep_w5, "pep")
        })

        # Negative examples: mismatched epitopes.
        neg_eps = np.random.choice(
            epitopes[epitopes != r[epitope_col]],
            size=min(3, len(epitopes)-1),
            replace=False
        )

        for e in neg_eps:
            e_w5 = center5(e)
            if e_w5 is None:
                continue
            rows.append({
                "cdr3b": r["cdr3b"],
                "epitope": e,
                "label": 0,
                "tcr_feats": featurize_window(w5, "tcr"),
                "pep_feats": featurize_window(e_w5, "pep")
            })

    #Build feature matrix.
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

    #Epitope-held-out split.
    unique_eps = np.unique(epitope_ids)

    train_eps, test_eps = train_test_split(
        unique_eps, test_size=0.2, random_state=0
    )

    train_mask = np.isin(epitope_ids, train_eps)
    test_mask  = np.isin(epitope_ids, test_eps)

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    #Train interpretable model.
    clf = DecisionTreeClassifier(
        max_depth=20,
        min_samples_leaf=2,
        random_state=0
    )

    clf.fit(X_train, y_train)

    #Evaluate on unseen epitopes.
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:,1]

    print("Unseen-epitope accuracy:", accuracy_score(y_test, y_pred))

    if VISUALIZE_TREE:
        import matplotlib.pyplot as plt
        from sklearn.tree import plot_tree

        plt.figure(figsize=(50, 20))  # Taller layout for readability.
        plot_tree(
            clf,
            feature_names=X.columns,
            class_names=["nonbinder", "binder"],
            filled=True,
            rounded=True,
            proportion=True,
            fontsize=20,
        )
        plt.title("Decision Tree Classifier for CDR3 Binding (Vertical Layout)", fontsize=16)
        plt.savefig("decision_tree.png", dpi=300, bbox_inches="tight")
        plt.show()

    if SHOW_CONFUSION_MATRIX:
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        import seaborn as sns

        cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["nonbinder", "binder"],
            yticklabels=["nonbinder", "binder"],
            ax=ax,
        )
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
        plt.show()

    feature_names = list(X.columns)
    rules = extract_tree_rules(clf, feature_names, class_names=["nonbinder","binder"], target_class=1)

    def binder_prob(rule):
        proba = rule["proba"]
        return float(proba[1]) if len(proba) > 1 else float(proba[0])

    sorted_rules = sorted(
        list(enumerate(rules, 1)),
        key=lambda ir: (binder_prob(ir[1]), ir[1]["n_samples"]),
        reverse=True,
    )

    # Peptide of interest
    if peptide is None:
        peptide = DEFAULT_PEPTIDE
    unseen_pep_core = center5(peptide)
    if unseen_pep_core is None:
        raise ValueError("Peptide must be at least 5 amino acids long.")

    fixed_pep = peptide_fixed_assignment(
        unseen_pep_core,
        all_feature_names=list(X.columns),
        plus=plus, minus=minus, hydro=hydro, polar=polar
    )

    clauses_tcr = conditioned_tree_to_dnf_over_tcr(clf, list(X.columns), fixed_pep, positive_class=1)
    clauses_tcr = apply_same_anchor_rule(clauses_tcr)
    formula_tcr_only = dnf_to_string(clauses_tcr)

    if SHOW_TREE_TEXT:
        print("\nRule summary:")
        print(f"Total binder rules: {len(rules)}")
        print(f"Showing top {min(MAX_RULES_TO_PRINT, len(rules))} rules by binder prob, then n.")

        for i, r in sorted_rules[:MAX_RULES_TO_PRINT]:
            conditions = [cond_to_readable(f, op, t) for f, op, t in r["conditions"]]
            print(f"\nRule {i}: literals={len(conditions)}, n={r['n_samples']}, p(binder)={binder_prob(r):.3f}")
            for cond in conditions:
                print("  ", cond)

    dnf = rules_to_dnf(rules)
    print(f"\nDNF: clauses={len(rules)}, length={len(dnf)}")
    print(dnf[:200] + ("..." if len(dnf) > 200 else ""))

    print("\nPeptide-conditioned DNF:")
    print(f"clauses={len(clauses_tcr)}, length={len(formula_tcr_only)}")
    print(formula_tcr_only[:200] + ("..." if len(formula_tcr_only) > 200 else ""))

    #Run simulator and compare to brute force.
    clauses = clauses_for_peptide(peptide)
    print(f"[clauses] Using {len(clauses)} clauses for peptide={peptide}")

    expanded_clauses = expand_clauses_for_oracle(clauses)
    M, sols = brute_force_solutions(expanded_clauses, limit_print=8)
    N = 2**10

    # Grover iteration heuristic.
    iters = max(1, int(round((math.pi / 4) * math.sqrt(N / max(1, M)))))
    print(f"[grover] Using iters = {iters} (based on M={M})")

    qc = build_grover_circuit_from_clauses(expanded_clauses, iters=iters)
    predicate = lambda data10: predicate_from_clauses(data10, expanded_clauses)
    print("[grover] Circuit qubits:", qc.num_qubits)

    backend = Aer.get_backend("aer_simulator")
    tqc = transpile(qc, backend=backend, optimization_level=1, seed_transpiler=SEED)
    
    num_shots = 5000
    res = backend.run(tqc, shots=num_shots, seed_simulator=SEED).result()
    counts = res.get_counts()

    def key_to_c0_to_c11(key: str) -> str:
        """Convert Qiskit c11..c0 output to c0..c11 order."""
        return key[::-1]

    def split_data_and_checks(key: str) -> Tuple[str, str]:
        """Split a measurement key into data bits and check bits."""
        c = key_to_c0_to_c11(key)
        data10 = c[:10]       # c0..c9 corresponds to q0..q9 measurements
        checks = c[10:12]     # c10..c11 corresponds to q10,q11 checks
        return data10, checks

    # Raw valid rate (ignoring checks).
    valid_shots = 0
    for key, ct in counts.items():
        data10, _ = split_data_and_checks(key)
        if predicate(data10):
            valid_shots += ct
    print(f"\nVALID shots (raw) = {valid_shots} / {num_shots}  ({valid_shots/num_shots:.3f})")

    # Postselected valid rate (keep only checks==00).
    kept_shots = 0
    valid_kept_shots = 0
    for key, ct in counts.items():
        data10, checks = split_data_and_checks(key)
        if checks == "00":
            kept_shots += ct
            if predicate(data10):
                valid_kept_shots += ct

    if kept_shots > 0:
        print(f"Postselected on checks==00: kept = {kept_shots} / {num_shots}  ({kept_shots/num_shots:.3f})")
        print(f"VALID among kept = {valid_kept_shots} / {kept_shots}  ({valid_kept_shots/kept_shots:.3f})")
    else:
        print("Postselected on checks==00: kept = 0 (no shots passed checks)")

    shots = sum(counts.values())
    baseline = shots / (2**10)
    threshold = baseline * 2.0
    valid_heavy = []
    for key, ct in counts.items():
        data10, _ = split_data_and_checks(key)
        if predicate(data10) and ct >= threshold:
            valid_heavy.append(data10)
    print(f"Estimated #solutions (valid & >= {threshold:.1f} counts): {len(set(valid_heavy))}")

    aa_validation = pd.read_csv("./aa_table.csv", index_col=0)
    aa_map_short_to_1_letter = {
        'Cys': 'C', 'Met': 'M', 'Phe': 'F', 'Ile': 'I', 'Leu': 'L', 'Val': 'V',
        'Trp': 'W', 'Tyr': 'Y', 'Ala': 'A', 'Gly': 'G', 'Thr': 'T', 'Ser': 'S',
        'Asn': 'N', 'Gln': 'Q', 'Asp': 'D', 'Glu': 'E', 'His': 'H', 'Arg': 'R',
        'Lys': 'K', 'Pro': 'P'
    }

    aa_validation_1_letter = aa_validation.rename(index=aa_map_short_to_1_letter, columns=aa_map_short_to_1_letter)

    aa_categories = pd.DataFrame({
        'Amino Acid': aa_validation.index,
        'Category': [aa2grp(aa_map_short_to_1_letter[aa]) for aa in aa_validation.index]
    })

    # Map single-letter amino acids to their categories.
    single_letter_to_category_map = {
        aa_map_short_to_1_letter[aa_name]: category
        for aa_name, category in zip(aa_categories['Amino Acid'], aa_categories['Category'])
    }

    # Define the categories.
    categories = ['H', 'P', '+', '-']

    # Initialize a category-averaged interaction matrix.
    category_avg_matrix = pd.DataFrame(index=categories, columns=categories, dtype=float)

    # Populate the matrix by averaging interaction energies per category pair.
    for row_cat in categories:
        # Get all single-letter AAs belonging to the current row category.
        row_aas = [aa_char for aa_char, cat in single_letter_to_category_map.items() if cat == row_cat]

        for col_cat in categories:
            # Get all single-letter AAs belonging to the current column category.
            col_aas = [aa_char for aa_char, cat in single_letter_to_category_map.items() if cat == col_cat]

            # Ensure there are amino acids for both categories before slicing.
            if row_aas and col_aas:
                # Extract the sub-matrix of interaction energies.
                sub_matrix = aa_validation_1_letter.loc[row_aas, col_aas]

                # Calculate the average and store it.
                category_avg_matrix.loc[row_cat, col_cat] = sub_matrix.mean().mean()
            else:
                # If no amino acids exist for a category, set NaN.
                category_avg_matrix.loc[row_cat, col_cat] = float('nan')
    category_avg_matrix_abs= np.abs(category_avg_matrix)

    start_index = (len(peptide) - 5) // 2
    end_index = start_index + 5

    # Extract the 5-amino acid core.
    peptide_5_core = peptide[start_index:end_index]
    print(f"Original peptide: {peptide}")
    print(f"Extracted 5-amino acid core: {peptide_5_core}")

    # Convert the peptide core to categories.
    peptide_5_core_categories = [aa2grp(aa) for aa in peptide_5_core]

    def score_solution(solution_bits: str, peptide_core_categories: List[str]) -> float:
        """Score a solution by category interaction energy."""
        decoded_categories = decode_10bit_to_classes(solution_bits)
        score = 0.0
        for i in range(5):
            sol_cat = decoded_categories[i]
            core_cat = peptide_core_categories[i]
            # Use the absolute interaction energy for scoring.
            interaction_energy = category_avg_matrix_abs.loc[sol_cat, core_cat]
            score += interaction_energy
        return score

    top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:15]
    print("\nTop measured results (showing check bits):")
    for key, ct in top:
        data10, checks = split_data_and_checks(key)
        ok = predicate(data10)
        chk_ok = (checks == "00")
        tag = "VALID+CHK" if (ok and chk_ok) else ("VALID" if ok else ("CHK" if chk_ok else ""))
        bio_score = score_solution(data10, peptide_5_core_categories)
        print(
            f"{key}  {ct:4d}  q0..q9={data10}  checks(c10,c11)={checks}  "
            f"{decode_10bit_to_classes(data10)}  {tag}  BIO={bio_score:.3f}"
        )

    # Select the solution with highest count and highest bio score.
    valid_checked_solutions = []
    for key, ct in counts.items():
        data10, checks = split_data_and_checks(key)
        if predicate(data10) and checks == "00":
            bio_score = score_solution(data10, peptide_5_core_categories)
            valid_checked_solutions.append((data10, ct, bio_score))
    if valid_checked_solutions:
        best_solution = max(valid_checked_solutions, key=lambda x: (x[1], x[2]))
        best_data10, best_count, best_bio_score = best_solution
        best_sol_classes = decode_10bit_to_classes(best_data10)
        print(
            f"\nBest valid+checked solution: q0..q9={best_data10}  "
            f"{best_sol_classes}  count={best_count}  BIO={best_bio_score:.3f}"
        )
    else:
        print("\nNo valid+checked solutions found; skipping candidate expansion.")
        return

    groups = {
    "+": list("KRH"),
    "-": list("DE"),
    "H": list("AVILMFWY"),
    "P": list("STNQGPC")}

    from itertools import product

    def expand_groups(group_string):
        """Return all amino-acid strings consistent with a group pattern."""
        choices = [groups[g] for g in group_string]
        for combo in product(*choices):
            yield "".join(combo)
    candidates = list(expand_groups(best_sol_classes))
    print(f"\nTotal candidate sequences from best solution: {len(candidates)}")

    # Score candidates using the full amino-acid interaction table.
    aa_validation_abs = np.abs(aa_validation_1_letter)
    def score_candidate(candidate: str, peptide_core: str) -> float:
        """Score a concrete amino-acid candidate against the peptide core."""
        score = 0.0
        for i in range(5):
            aa_sol = candidate[i]
            aa_core = peptide_core[i]
            interaction_energy = aa_validation_abs.loc[aa_sol, aa_core]
            score += interaction_energy
        return score
    scored_candidates = []
    for cand in candidates:
        bio_score = score_candidate(cand, peptide_5_core)
        scored_candidates.append((cand, bio_score))
    scored_candidates = sorted(scored_candidates, key=lambda x: x[1], reverse=True)
    print("\nTop candidate sequences by bio score:")
    for cand, bio_score in scored_candidates[:10]:
        print(f"  {cand}  BIO={bio_score:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TCR/epitope rule extraction and Grover search.")
    parser.add_argument("--peptide", default=DEFAULT_PEPTIDE, help="Peptide sequence to analyze.")
    args = parser.parse_args()
    main(peptide=args.peptide)
  
    
