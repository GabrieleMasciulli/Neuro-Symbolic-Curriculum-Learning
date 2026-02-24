# Neuro-Symbolic Curriculum Learning

This repository extends the [RSBench]([https://github.com/ema-marconato/rsbench](https://github.com/unitn-sml/rsbench-code)) codebase with data-centric strategies—**Curriculum Learning** and **Active Learning**—to mitigate **Reasoning Shortcuts (RSs)** in Neuro-Symbolic (NeSy) architectures trained via weak supervision. Experiments are conducted on MNIST-Half, Shortcut-MNIST, and BDD-OIA.

---

## Key Additions

### `utils/class_specific_risk.py`

Computes **class-specific risk upper bounds** via constrained optimization (SLSQP). Given the observed partial risk of a model and the marginal concept distribution, it solves a quadratic program to find the worst-case misclassification rate for each concept class. Used as a difficulty metric for class-risk curriculum learning on MNIST-style datasets.

### `utils/instance_specific_risk.py`

Computes an **instance-specific risk** score for each training sample without requiring concept labels. For MNIST-Addition tasks it measures how much probability mass the model places on concept pairs that violate the symbolic constraint, normalized by the number of valid pairs. Used for instance-risk curriculum and active learning sample selection.

### `utils/risk_curriculum_sampler.py`

Implements `ClassSpecificRiskCurriculumSampler` and `InstanceSpecificRiskCurriculumSampler`, PyTorch `Sampler` subclasses that progressively expose training data ordered from easiest to hardest according to the risk scores above. The active fraction of data is controlled by `current_phase` (0→1 over training).

### `utils/train.py`

Extended the standard training loop with:

- **Curriculum learning** support: integrates the risk-based samplers and updates the active data fraction across curriculum steps.
- Periodic **risk recomputation** during training (`--risk_update_freq`) to dynamically refresh difficulty scores as the model improves.
- Concept-coverage diagnostics printed per epoch when curriculum is active.

### `utils/train_active.py`

Implements a full **Active Learning (AL)** training loop:

- Starts with zero concept supervision; at each AL cycle selects `--al_query_size` samples from the unlabeled pool and grants them concept supervision.
- **Random strategy** (`--active_type random`): uniform random selection.
- **Instance-risk strategy** (`--active_type instance_risk`): ranks unlabeled samples by instance-specific risk and queries the most ambiguous ones first, achieving faster RS breakdown with a minimal annotation budget.
- Model weights are reset to the initial state at the start of each cycle to avoid catastrophic interference.

---

## Main Findings

| Strategy                         | MNIST-Half / Shortcut-MNIST                                    | BDD-OIA                         |
| -------------------------------- | -------------------------------------------------------------- | ------------------------------- |
| Curriculum (class/instance risk) | ❌ Increases RS formation by reducing constraint diversity     | ✅ Marginal gains in concept F1 |
| Active Learning (instance risk)  | ✅ Faster convergence vs. random; breaks RSs with fewer labels | —                               |

Curriculum learning counter-intuitively _worsens_ RSs on weakly supervised arithmetic tasks by restricting early-training distribution diversity. Conversely, using instance-specific risk for **active query selection** efficiently breaks RSs with a small supervision budget.

---
