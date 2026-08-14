# Input Format

`run_experiment.py` expects two directory paths:

- `--lkn-obs-dir`: local knowledge networks available in the observation period.
- `--lkn-pred-dir`: local knowledge networks available in the future prediction period.

Each directory must contain matching `<scholar_id>.txt` files. Each nonempty line is parsed as two whitespace-separated topic identifiers. Repeated edges and reversed edges are permitted. Lines with fewer than two fields are ignored.

The GKN vocabulary and edges are built from observation-window files only. Topics first observed in the prediction window are not added to the vocabulary, which prevents future-information leakage into graph construction.

For every contributing scholar, positive candidates are topics that appear in the prediction window but not the observation window. The script samples one negative candidate per positive candidate from topics absent from both windows for that scholar.

The default filter is empty. Supply `--excluded-topics-file` when generic or out-of-scope topics should be removed. The file contains one topic ID per line; blank lines and lines starting with `#` are ignored.

The input files, topic maps, and generated manifests are intentionally excluded from version control because they can encode identifiable research histories.
