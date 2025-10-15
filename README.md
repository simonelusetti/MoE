# Expert MoE Rationale Model

## Token-to-Factor Mixture-of-Experts

Let a sentence be a token sequence
```
x = (t_1, ..., t_T)
```
with contextual states `h_i = Enc(x)_i` stacked into `H = [h_1, ..., h_T]^T`.

We posit `K` latent factors (experts). A gating network produces token-to-factor probabilities
```
pi_i = softmax(W_g h_i)
```
with rows `pi_i` forming `Pi` (`T x K`). Each expert aggregates tokens
```
z_k = sum_i pi_{ik} h_i
```
and can be transformed (`tilde z_k = f_k(z_k)`).

Sentence reconstruction via factor projections `V_k` yields
```
hat e(x) = sum_k V_k tilde z_k
```
where `hat e(x)` matches a target embedding `e(x)`.

### Losses
- Sentence reconstruction: `L_sent = ||e(x) - hat e(x)||_2^2`
- Token reconstruction: `hat h_i = g(sum_k pi_{ik} tilde z_k)`, `L_tok = sum_i ||h_i - hat h_i||_2^2`
- Entropy regulariser: `L_ent = sum_i H(pi_i)`
- Overlap penalty: `L_overlap = (1/T) sum_i sum_{k<j} pi_{ik} pi_{ij}`
- Diversity: `L_div = ||Z^T Z - I||_F^2`, `Z = [tilde z_1, ..., tilde z_K]`
- Load balancing: `u_k = (1/T) sum_i pi_{ik}`, `L_bal = sum_k (u_k - 1/K)^2`

Combined objective
```
L = λ_sent L_sent + λ_tok L_tok + λ_ent L_ent + λ_overlap L_overlap + λ_div L_div + λ_bal L_bal
```
with non-negative weights `λ_*`.

### Outputs
- Token-to-factor assignments `Pi`
- Expert embeddings `tilde z_k`
- Per-factor precision/recall/F1 (when labels exist)
