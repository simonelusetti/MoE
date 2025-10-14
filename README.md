## Expert MoE Rationale Model

This directory hosts a stand-alone training script for the mixture-of-experts rationale selector.
It mirrors the layout of `RatCon`:

- `expert_moe/models.py` implements the expert selector.
- `expert_moe/train.py` provides a Hydra entry-point that trains or evaluates the model.
- `expert_moe/conf/default.yaml` stores default hyperparameters.

Run training via Dora (from the repository root):

```bash
dora -P expert_moe.train run
```

Override configuration values using Hydra-style arguments, for example:

```bash
dora -P expert_moe.train run model.expert.num_experts=6 train.epochs=20
```
