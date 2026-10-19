
# What Can Labels Alone Audit and Edit in Frozen Representations? — Reproduction Package

This repository contains the code for the paper
*"What Can Labels Alone Audit and Edit in Frozen Representations?"* .

## Repository layout

| File | Purpose |
|---|---|
| `run_labelonly_workflow_full_v2_appendix.py` | Core implementation (operators, DMD fitting, probes) + full runner including appendix experiments and synthetic illustrations. Imported as a base module by the rebuttal scripts. |
| `run_labelonly_workflow_full_v1_multiseed.py` | Multi-seed runner for the main-text experiments (10 outer seeds): Table 1, Table 2, and Figure 1–4 inputs. |
| `run_labelonly_workflow_full_multiseed.ps1` / `run_labelonly_workflow_full_appendix_multiseed.ps1` | PowerShell launchers (Windows); equivalent shell commands map one-to-one onto the argparse flags. |
| `extract_feature_cache_clip.py` | Cache frozen CLIP ViT-B/32 features for CelebA / Waterbirds / PACS. |
| `run_labelonly_rebuttal_experiments.py` (+ `.ps1`) | Rebuttal experiments: stage-1 equivalence (DMD vs LDA/PLS), protected-deletion overlap, quotient/residual leakage decomposition, budgeted q-frontiers, alternative auditor probes (linear/quadratic/MLP), r/α sensitivity. |
| `run_rebuttal_q8_paired_bootstrap.py` | Two-level paired bootstrap CIs for the fixed-rank (q=8) editing results on the six vision settings. |
| `run_civilcomments_balanced_audit.py` | CivilComments (frozen text representations): MLP auditor reduction, task retention, and paired bootstrap CIs. |
| `build_civilcomments_frozen_cache.py` | Build CivilComments caches with sentence encoders (all-mpnet-base-v2, BGE-large-en-v1.5): fixed 40k/10k cohort, toxicity label, lexicon identity-mention attribute. |
| `build_civilcomments_LLM_decoder_cache.py` | Build CivilComments caches with decoder-only LLMs (last non-padding token hidden state, e.g. Qwen2.5-1.5B). |
| `run_rebuttal_build_waterbirds_vlm_cache.py` | Cache frozen image features for larger vision backbones (OpenCLIP ViT-L/14). |
| `run_labelonly_rebuttal_frozen_exp2.ps1` | End-to-end CivilComments pipeline launcher (core 10-seed runs). |
| `summarize_rebuttal_results.py` | Aggregate `all_rebuttal_records.csv` files into compact summary tables. |

## Environment
Python >= 3.10
numpy, pandas, scikit-learn, matplotlib
torch, transformers, datasets, sentence-transformers, Pillow, tqdm, open_clip_torch, torchvision

## License

MIT (code). Datasets and pretrained backbones retain their own licenses.


