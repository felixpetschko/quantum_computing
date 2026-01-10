# Quantum Computing Project Description

## Biological Use Case

T-cells are a crucial part of the immune system and they can detect cancer cells and pathogens. They recognize disease-associated signals through T-cell receptors (TCRs) on their surface, which bind to short peptide antigens presented by major histocompatibility complex (MHC) molecules. A very important part of the T-cell receptor (TCR) is the CDR3 region that is a sequence of around 15 amino acids (20 different amino acids possible per position). This region contributes most strongly to antigen specificity and largely determines which peptide–MHC complexes a TCR can recognize.

TCR-engineered T-cells are genetically modified T-cells that express a selected or engineered TCR so that they can recognize disease-related antigens, e.g. cancer antigens. To generate such cells it is essential to identify their TCR sequence, especially that in the CDR3 region that can bind a given peptide-MHC complex. The goal of our project is to use quantum computing to find binding candidate CDR3 sequences for a given antigen peptide (short chain of amino acids). 


## Summary of Approach

First, we load a database of known T-cell receptor sequences with the according antigens they bind to. We divide the amino acids of the sequences into 4 classes based on their physical properties ( [H] - hydrophobic, [P] - polar, [+] - positively charged, [-] - negatively charged). That way we only have 4 possible values per position instead of 20 (number of different amino acids). Then we train a decision tree that should learn to classify binders and non-binders. Based on the decision tree, we derive boolean rules that should hold for a sequence that binds to a specific antigen (e.g. position 2 of the CDR3 sequence should be hydrophobic or polar and position 3 should not be negatively charged or ...). We construct a phase oracle with the boolean rules and use grover’s search to find patterns for a binding CDR3 sequence (e.g. ['H', '-', 'P', 'P', 'H'] ). Lastly, we generate the possible amino acid sequences based on the pattern we found and choose the best sequence based on a biological score that is computed with the known binding properties of the amino acids (interaction-energy tables).

Due to the quantum hardware limitations, we only consider the 5 core positions of the sequences. Our assumption is that the binding rules for longer amino acid sequences that can be learned with some machine learning approach or that are given by some biology expert might give us a satisfiability problem that is NP-hard and could take very long to solve. In that case, quantum computing could give quadratic speedup compared to classical hardware. With this project, we showcase a full example pipeline from a database of known binding sequences to a candidate solution of a CDR3 sequence that recognizes an unseen antigen. Finding such binding sequences would solve an important problem in TCR T-cell engineering.


## HowTo
The necessary libraries can be installed into a conda environment with the according .yml environment file. The code (quantum_project.py) was tested with the Qiskit simulator on the Leo5 HPC cluster.
You can run the code with:

```bash
python quantum_project.py --peptide GILVAMTFC
```

The --peptide input parameter specifies the peptide for which we want to find a binding CDR3 sequence.
For many inputs you can't find binding sequences. For testing, one could try the following inputs for which we found solutions:

- GILVAMTFC
- DVWQKSLTM
- FVGKLMHA
- GLCTLVAMV
- KELGHTVAP
- DEKRHQLMV
- ATVGLMPHR
- QWERTYVSL
- MVACTRLGH
- KHVLTAGMR
