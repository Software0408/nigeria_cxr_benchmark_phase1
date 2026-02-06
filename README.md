# nigeria_cxr_benchmark_phase1
Phase 1 foundations for Nigerian/African CXR benchmark – evaluation pipeline only

The anonymization salt is stored in an encrypted environment under PI control and is not distributed with the dataset

### Label Hierarchy
The project uses a clinically inspired 3-tier label system defined in `configs/labels/hierarchy.yaml` and implemented in `src/nlp/labels.py`.
- Tier 1: Clinical state (normal / abnormal / indeterminate)
- Tier 2: Pathology categories (for subgroup analysis)
- Tier 3: Specific findings from Appendix A (multi-label)