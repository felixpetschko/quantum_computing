# Quantum Computing Project Description

## Biological Use Case

T-cells are a crucial part of the immune system and they can detect cancer cells and pathogens. They do that with receptors at their surface that bind to certain types of molecules called antigens. A very important part of the T-cell receptor (TCR) is the CDR3 region that is a sequence of around 15 amino acids (20 different amino acids possible per position). This sequence mainly determines which antigens a receptor can bind to.

TCR-engineered T-cells are genetically modified T-cells with a specifically designed TCR so that they can recognize disease-related antigens, e.g. cancer antigens. To create these cells we need to know which CDR3 sequence would bind to which antigen. The goal of our project is to use quantum computing to find binding CDR3 sequences for a given antigen peptide (short chain of amino acids). 


## Summary of Approach

First, we load a database of known T-cell receptor sequences with the according antigens they bind to. We divide the amino acids of the sequences into 4 classes based on their physical properties ( [H] - hydrophobic, [P] - polar, [+] - positively charged, [-] - negatively charged). That way we only have 4 possible values per position instead of 20 (number of different amino acids). Then we train a decision tree that should learn to classify binders and non-binders. Based on the decision tree, we derive boolean rules that should hold for a sequence that binds to a specific antigen (e.g. position 2 of the CDR3 sequence should be hydrophobic or polar and position 3 should not be negatively charged and so on). We construct a phase oracle with the boolean rules and use grover’s search to find patterns for a binding CDR3 sequence (e.g. ['H', '-', 'P', 'P', 'H'] ). Lastly, we generate the possible amino acid sequences based on the pattern we found and choose the best sequence based on a biological score that is computed with the known binding properties of the amino acids (interaction-energy tables).

Due to the quantum hardware limitations, we only consider the 5 core positions of the sequences. Our assumption is that the binding rules for longer amino acid sequence that can be learned with some machine learning approach or that are given by some biology expert might give us a satisfiability problem that is NP-hard and could take very long to solve. In that case, quantum computing could give quadratic speedup compared to classical hardware. With this project we showcase a full example pipeline from a database of known binding sequences to a candidate solution of a CDR3 sequence that recognizes an unseen antigen. Finding such binding sequences would solve an important problem in TCR T-cell engineering.

The necessary libraries can be installed into a conda environment with the according .yml environment file. The code (quantum_project.py) was tested with the Qiskit simulator on the Leo5 HPC cluster. 

