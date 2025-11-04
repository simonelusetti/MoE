import torch
import torch.nn as nn
import torch.nn.functional as F

from sentence_transformers import SentenceTransformer


class ExpertModel(nn.Module):
    """
    Token-to-factor mixture-of-experts rationale model.
    Routes each token to latent experts, aggregates expert embeddings,
    and reconstructs the anchor sentence embedding (and optionally token states).
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        expert_cfg = cfg.expert

        self.num_experts = int(expert_cfg.num_experts)
        if self.num_experts < 1:
            raise ValueError("cfg.model.expert.num_experts must be >= 1")

        self.routing = expert_cfg.routing.lower()
        self.routing_tau = float(expert_cfg.routing_tau)
        self.normalize_factors = bool(expert_cfg.normalize)
        self.dropout = float(expert_cfg.dropout)
        self.use_balance = expert_cfg.use_balance
        self.use_diversity = expert_cfg.use_diversity

        self.sbert = SentenceTransformer(cfg.sbert_name)
        self.pooler = self.sbert[1]
        enc_hidden = self.sbert[0].auto_model.config.hidden_size

        factor_dim = int(expert_cfg.factor_dim)
        factor_hidden = int(expert_cfg.factor_hidden)
        self.factor_dim = factor_dim

        gate_hidden = int(expert_cfg.gate_hidden)
        gate_dropout = float(expert_cfg.gate_dropout)
        gate_layers = []
        in_dim = enc_hidden
        if gate_hidden > 0:
            gate_layers.append(nn.Linear(in_dim, gate_hidden))
            gate_layers.append(nn.GELU())
            if gate_dropout > 0.0:
                gate_layers.append(nn.Dropout(gate_dropout))
            in_dim = gate_hidden
        gate_layers.append(nn.Linear(in_dim, self.num_experts))
        self.gate = nn.Sequential(*gate_layers)

        use_shared_transform = bool(expert_cfg.shared_transform)
        transform_dropout = float(expert_cfg.transform_dropout)
        if use_shared_transform:
            self.expert_transforms = None
            self.shared_transform = self._build_transform(
                enc_hidden, factor_hidden, factor_dim, transform_dropout
            )
        else:
            self.shared_transform = None
            modules = []
            for _ in range(self.num_experts):
                modules.append(
                    self._build_transform(enc_hidden, factor_hidden, factor_dim, transform_dropout)
                )
            self.expert_transforms = nn.ModuleList(modules)

        sentence_dim = self.pooler.get_sentence_embedding_dimension()
        recon_modules = []
        recon_hidden = int(expert_cfg.reconstruction_hidden)
        for _ in range(self.num_experts):
            layers = []
            in_features = factor_dim
            if recon_hidden > 0:
                layers.append(nn.Linear(in_features, recon_hidden))
                layers.append(nn.GELU())
                if transform_dropout > 0:
                    layers.append(nn.Dropout(transform_dropout))
                in_features = recon_hidden
            layers.append(nn.Linear(in_features, sentence_dim))
            recon_modules.append(nn.Sequential(*layers))
        self.reconstruction_heads = nn.ModuleList(recon_modules)

        self.use_token_decoder = bool(expert_cfg.use_token_decoder)
        if self.use_token_decoder:
            decoder_hidden = int(expert_cfg.decoder_hidden)
            decoder_layers = []
            decoder_in = factor_dim
            if decoder_hidden > 0:
                decoder_layers.append(nn.Linear(decoder_in, decoder_hidden))
                decoder_layers.append(nn.GELU())
                if transform_dropout > 0:
                    decoder_layers.append(nn.Dropout(transform_dropout))
                decoder_in = decoder_hidden
            decoder_layers.append(nn.Linear(decoder_in, enc_hidden))
            self.token_decoder = nn.Sequential(*decoder_layers)
        else:
            self.token_decoder = None

        self.small_value = 1e-6

    @staticmethod
    def _build_transform(in_dim, hidden_dim, out_dim, dropout):
        layers = []
        if hidden_dim > 0:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_dim, out_dim))
        else:
            layers.append(nn.Linear(in_dim, out_dim))
        return nn.Sequential(*layers)

    def _apply_routing(self, logits, mask):
        mask = mask.unsqueeze(-1).to(logits.dtype)
        logits = logits.masked_fill(mask == 0, -1e9)

        if self.routing == "gumbel":
            routed = F.gumbel_softmax(logits, tau=self.routing_tau, hard=False, dim=-1)
        elif self.routing == "softmax":
            routed = torch.softmax(logits, dim=-1)
        else:
            raise ValueError(f"Unsupported routing '{self.routing}'. Use 'softmax' or 'gumbel'.")

        routed = routed * mask
        routed_sum = routed.sum(dim=-1, keepdim=True).clamp_min(self.small_value)
        routed = routed / routed_sum
        return routed

    def _transform_factors(self, factors):
        if self.shared_transform is not None:
            flat = factors.reshape(-1, factors.size(-1))
            transformed = self.shared_transform(flat)
            return transformed.view(factors.size(0), self.num_experts, -1)

        outputs = []
        for idx, module in enumerate(self.expert_transforms):
            outputs.append(module(factors[:, idx, :]))
        return torch.stack(outputs, dim=1)

    def _compute_diversity_penalty(self, factors):
        if factors.size(0) < 2:
            return factors.new_tensor(0.0)
        centered = factors - factors.mean(dim=0, keepdim=True)
        flat = centered.view(factors.size(0), self.num_experts, -1)
        gram = torch.einsum("bkd,bjd->kj", flat, flat) / flat.size(0)
        diag = torch.diag(gram)
        off_diag = gram - torch.diag_embed(diag)
        return off_diag.pow(2).sum()

    def forward(self, embeddings, attention_mask, incoming=None, outgoing=None):
        mask_float = attention_mask.to(dtype=embeddings.dtype)
        gate_logits = self.gate(embeddings)
        routing_weights = self._apply_routing(gate_logits, mask_float)

        factors_raw = torch.einsum("btk,btd->bkd", routing_weights, embeddings)
        mass = routing_weights.sum(dim=1).clamp_min(self.small_value).unsqueeze(-1)
        if self.normalize_factors:
            factors_raw = factors_raw / mass
        transformed_factors = self._transform_factors(factors_raw)

        pooled = self.pooler(
            {
                "token_embeddings": embeddings,
                "attention_mask": attention_mask,
            }
        )["sentence_embedding"]

        recon_parts = []
        for idx, head in enumerate(self.reconstruction_heads):
            recon_parts.append(head(transformed_factors[:, idx, :]))
        reconstruction = torch.stack(recon_parts, dim=1).sum(dim=1)

        token_reconstruction = None
        if self.token_decoder is not None:
            mixture = torch.einsum("btk,bkf->btf", routing_weights, transformed_factors)
            token_reconstruction = self.token_decoder(mixture)

        entropy = -(routing_weights.clamp_min(self.small_value).log() * routing_weights)
        entropy = (entropy.sum(dim=-1) * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp_min(1.0)

        pi_sq = (routing_weights ** 2).sum(dim=-1)
        overlap = 0.5 * (1.0 - pi_sq)
        overlap = (overlap * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp_min(1.0)

        if self.use_balance:
            expert_mass = routing_weights.sum(dim=1)
            total_tokens = mask_float.sum(dim=1, keepdim=True).clamp_min(1.0)
            balanced_mass = expert_mass / total_tokens
            target = routing_weights.new_full((1, self.num_experts), 1.0 / self.num_experts)
            balance = ((balanced_mass.mean(dim=0, keepdim=True) - target) ** 2).sum()
        else:
            balance = routing_weights.new_zeros(())

        if self.use_diversity:
            diversity = self._compute_diversity_penalty(transformed_factors)
        else:
            diversity = routing_weights.new_zeros(())

        return {
            "pi": routing_weights,
            "factors_raw": factors_raw,
            "factors": transformed_factors,
            "anchor": pooled,
            "reconstruction": reconstruction,
            "token_reconstruction": token_reconstruction,
            "entropy": entropy,
            "overlap": overlap,
            "balance": balance,
            "diversity": diversity,
        }


class ProductProjector(nn.Module):
    """Feed-forward projection module used to isolate latent subspaces."""

    def __init__(self, latent_dim: int, hidden_dim: int):
        super().__init__()
        layers = []
        if hidden_dim > 0:
            layers.append(nn.Linear(latent_dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, latent_dim))
        else:
            layers.append(nn.Linear(latent_dim, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ProductManifoldModel(nn.Module):
    """Token-level product-manifold projection module."""

    def __init__(self, cfg, input_dim: int):
        super().__init__()
        self.cfg = cfg
        model_cfg = cfg.product_model
        latent_dim = int(model_cfg.latent_dim)
        hidden = int(model_cfg.encoder_hidden)
        projector_hidden = int(model_cfg.projector_hidden)
        self.num_subspaces = int(model_cfg.num_subspaces)

        if self.num_subspaces < 1:
            raise ValueError("product_model.num_subspaces must be >= 1")

        self.latent_dim = latent_dim
        self.encoder = self._build_mlp(input_dim, hidden, latent_dim)
        self.token_decoder = self._build_mlp(latent_dim, hidden, input_dim)
        self.projectors = nn.ModuleList(
            ProductProjector(latent_dim, projector_hidden) for _ in range(self.num_subspaces)
        )

    @staticmethod
    def _build_mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Module:
        layers = []
        if hidden_dim > 0:
            layers.extend(
                [
                    nn.Linear(input_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, output_dim),
                ]
            )
        else:
            layers.append(nn.Linear(input_dim, output_dim))
        return nn.Sequential(*layers)

    def _encode_latents(
        self, token_embeddings: torch.Tensor, mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        z_hat = self.encoder(token_embeddings)
        projected = [proj(z_hat) for proj in self.projectors]
        subspaces = torch.stack(projected, dim=2)  # [B, T, K, D]
        aggregated = subspaces.sum(dim=2)  # [B, T, D]
        aggregated = aggregated * mask.unsqueeze(-1).type_as(aggregated)

        mask_float = mask.unsqueeze(-1).unsqueeze(-1).type_as(subspaces)
        masked_subspaces = subspaces * mask_float
        token_counts = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        factor_means = masked_subspaces.sum(dim=1) / token_counts.unsqueeze(-1)

        return {
            "z_hat": z_hat,
            "subspaces": subspaces,
            "aggregated_tokens": aggregated,
            "factors": factor_means,
        }

    def decode_tokens(self, aggregated_tokens: torch.Tensor) -> torch.Tensor:
        return self.token_decoder(aggregated_tokens)

    def forward(self, token_embeddings: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        latents = self._encode_latents(token_embeddings, mask)
        token_reconstruction = self.decode_tokens(latents["aggregated_tokens"])

        mask_float = mask.unsqueeze(-1).type_as(token_reconstruction)
        sentence_reconstruction = (
            (token_reconstruction * mask_float).sum(dim=1)
            / mask_float.sum(dim=1).clamp_min(1.0)
        )

        latents.update(
            {
                "token_reconstruction": token_reconstruction,
                "sentence_reconstruction": sentence_reconstruction,
                "mask": mask,
            }
        )
        return latents

    def encode_tokens(self, token_embeddings: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        return self._encode_latents(token_embeddings, mask)
