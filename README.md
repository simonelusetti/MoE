# Expert MoE Rationale Model

## Token-to-Factor Mixture-of-Experts

Let a sentence be a token sequence
\[
x = (t_1, \dots, t_T), \qquad T \in \mathbb{N},
\]
encoded by a contextualiser \(\mathrm{Enc}\) into
\[
H = [h_1, \dots, h_T]^\top \in \mathbb{R}^{T \times d}, \qquad h_i = \mathrm{Enc}(x)_i.
\]

We posit \(K\) latent factors (experts). A gating network produces token-to-factor probabilities
\[
\pi_i = \mathrm{Gate}(h_i) = \mathrm{softmax}(W_g h_i) \in \Delta^{K-1}, \qquad \pi_{ik} = p(k\mid t_i).
\]
Stacked over tokens: \(\Pi = [\pi_{ik}] \in \mathbb{R}^{T \times K}\).

Each expert aggregates the routed token states
\[
z_k = \sum_{i=1}^{T} \pi_{ik} \, h_i \in \mathbb{R}^{d},
\]
optionally processed by a small transform \( \tilde z_k = f_k(z_k; \theta_k) \in \mathbb{R}^{d_f} \).

The factors reconstruct a sentence embedding (e.g. SBERT target \(e(x)\)):
\[
\hat e(x) = \sum_{k=1}^{K} V_k \tilde z_k \in \mathbb{R}^{d_e}.
\]

### Losses

- **Sentence reconstruction**
  \[
  \mathcal{L}_{\text{sent}} = \|e(x) - \hat e(x)\|_2^2.
  \]
- **Token autoencoding (optional)**
  \[
  \hat h_i = g\!\Big(\sum_{k=1}^K \pi_{ik} \tilde z_k; \psi\Big), \qquad
  \mathcal{L}_{\text{tok}} = \sum_{i=1}^T \|h_i - \hat h_i\|_2^2.
  \]
- **Entropy regulariser** (encourage sharp routing)
  \[
  \mathcal{L}_{\text{ent}} = \sum_{i=1}^{T} H(\pi_i) = - \sum_{i,k} \pi_{ik} \log \pi_{ik}.
  \]
- **Overlap penalty** (discourage multi-expert selections of same token)
  \[
  \mathcal{L}_{\text{overlap}} = \frac{1}{T} \sum_{i=1}^{T} \sum_{k<j} \pi_{ik}\pi_{ij}.
  \]
- **Diversity penalty** (orthogonalise factor representations)
  \[
  \mathcal{L}_{\text{div}} = \big\|Z^\top Z - I\big\|_F^2, \quad Z = [\tilde z_1, \dots, \tilde z_K].
  \]
- **Load balancing** (equalise routing mass)
  \[
  u_k = \tfrac{1}{T}\sum_{i=1}^{T} \pi_{ik}, \qquad
  \mathcal{L}_{\text{bal}} = \sum_{k=1}^{K} \left(u_k - \tfrac{1}{K}\right)^2.
  \]

The total objective combines them with weights \(\lambda_\bullet\):
\[
\mathcal{L} =
\lambda_{\text{sent}} \mathcal{L}_{\text{sent}}
 + \lambda_{\text{tok}} \mathcal{L}_{\text{tok}}
 + \lambda_{\text{ent}} \mathcal{L}_{\text{ent}}
 + \lambda_{\text{overlap}} \mathcal{L}_{\text{overlap}}
 + \lambda_{\text{div}} \mathcal{L}_{\text{div}}
 + \lambda_{\text{bal}} \mathcal{L}_{\text{bal}}.
\]

### Outputs
- Token-to-factor assignments \(\Pi\) (interpretable rationales per expert).
- Expert embeddings \(\tilde z_k(x)\) capturing latent semantics.
- Per-factor metrics (precision/recall/F1) when gold token labels are available.
