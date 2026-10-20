"""
src/models/deberta_omni.py
===========================
Custom DeBERTa-v3-base Sentiment Classifier for Universal Review Intelligence.

Architecture upgrades over vanilla DistilBERT:
  1. Backbone:  microsoft/deberta-v3-base  (disentangled attention, RTD pre-training)
  2. Pooling:   Attention-Masked Mean + Max Pooling concatenation (captures late-paragraph complaints)
  3. Head:      5-Branch Multi-Sample Dropout (p=0.1..0.5) averaged before logits
  4. Output:    3-class (positive / neutral / negative)

Usage:
    from src.models.deberta_omni import DeBERTaOmniClassifier
    model = DeBERTaOmniClassifier(num_labels=3)
    outputs = model(input_ids, attention_mask)
    logits  = outputs["logits"]          # (B, 3)
    pooled  = outputs["pooled_output"]   # (B, 2*H)
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class AttentionMaskedPooling(nn.Module):
    """
    Concatenation of attention-masked Mean Pooling and Max Pooling
    over the full sequence (up to 512 tokens).

    Why:
      Standard [CLS] pooling compresses everything into the first token.
      Reviews with complaints at the end ("The food was good BUT the
      packaging was terrible") lose signal. Mean+Max pooling ensures
      every token contributes to the final representation.

    Output: (batch, 2 * hidden_size)
    """

    def forward(self, last_hidden_state: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        # last_hidden_state: (B, L, H)
        # attention_mask:    (B, L)
        mask_expanded = attention_mask.unsqueeze(-1).float()  # (B, L, 1)

        # Mean pooling (masked)
        sum_hidden = (last_hidden_state * mask_expanded).sum(dim=1)   # (B, H)
        sum_mask   = mask_expanded.sum(dim=1).clamp(min=1e-9)         # (B, 1)
        mean_pooled = sum_hidden / sum_mask                           # (B, H)

        # Max pooling (masked — fill padding with -inf before max)
        masked_hidden = last_hidden_state.clone()
        masked_hidden[attention_mask == 0] = -1e9
        max_pooled, _ = masked_hidden.max(dim=1)                     # (B, H)

        return torch.cat([mean_pooled, max_pooled], dim=-1)           # (B, 2H)


class MultiSampleDropoutHead(nn.Module):
    """
    5-Branch Multi-Sample Dropout Head.

    During training, runs the classification head through 5 dropout masks
    (p=0.1, 0.2, 0.3, 0.4, 0.5) and averages the logits. This is equivalent
    to a 5-model ensemble at zero extra inference cost:
      - At train time: averages 5 stochastic forward passes → smoother gradients
      - At eval time:  dropout is disabled → single deterministic forward pass

    Reference: Inoue (2019) "Multi-Sample Dropout for Accelerated Training
    and Better Generalization"
    """

    def __init__(self, input_dim: int, num_labels: int,
                 dropout_rates: tuple = (0.1, 0.2, 0.3, 0.4, 0.5)):
        super().__init__()
        self.dropouts   = nn.ModuleList([nn.Dropout(p) for p in dropout_rates])
        self.classifier = nn.Linear(input_dim, num_labels)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        if self.training:
            # Average logits across all dropout masks
            logits = torch.stack(
                [self.classifier(drop(pooled)) for drop in self.dropouts],
                dim=0,
            ).mean(dim=0)
        else:
            logits = self.classifier(pooled)
        return logits


class DeBERTaOmniClassifier(nn.Module):
    """
    Full custom DeBERTa-v3-base classifier.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID (default: microsoft/deberta-v3-base).
    num_labels : int
        Number of output classes (default: 3).
    dropout_rates : tuple
        Dropout probabilities for the multi-sample head.
    """

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",
        num_labels: int = 3,
        dropout_rates: tuple = (0.1, 0.2, 0.3, 0.4, 0.5),
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=self.config)
        self.pooler  = AttentionMaskedPooling()
        self.head    = MultiSampleDropoutHead(
            input_dim     = self.config.hidden_size * 2,   # Mean + Max concat
            num_labels    = num_labels,
            dropout_rates = dropout_rates,
        )
        self.num_labels = num_labels

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        encoder_out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # DeBERTa returns last_hidden_state at index 0
        hidden = encoder_out.last_hidden_state           # (B, L, H)
        pooled = self.pooler(hidden, attention_mask)      # (B, 2H)
        logits = self.head(pooled)                        # (B, num_labels)

        output = {"logits": logits, "pooled_output": pooled}

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            output["loss"] = loss_fn(logits, labels)

        return output

    def get_encoder_params_by_layer(self):
        """
        Returns a list of (layer_index, param_group) for Layer-Wise LR Decay.
        Layer 0 = embeddings (lowest LR), Layer N = top encoder layer (highest LR).
        """
        param_groups = []

        # Embeddings
        embed_params = []
        for name, param in self.encoder.named_parameters():
            if "embeddings" in name:
                embed_params.append(param)
        if embed_params:
            param_groups.append((-1, embed_params))  # -1 = embeddings layer

        # Encoder layers
        num_layers = self.config.num_hidden_layers
        for layer_idx in range(num_layers):
            layer_params = []
            for name, param in self.encoder.named_parameters():
                if f"layer.{layer_idx}." in name or f"layers.{layer_idx}." in name:
                    layer_params.append(param)
            if layer_params:
                param_groups.append((layer_idx, layer_params))

        return param_groups, num_layers

    def get_head_params(self):
        """Returns pooler + classification head parameters."""
        return list(self.pooler.parameters()) + list(self.head.parameters())


# ── Quick architecture summary ────────────────────────────────────────────
if __name__ == "__main__":
    model = DeBERTaOmniClassifier(num_labels=3)
    total   = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"DeBERTa-v3 Omni Classifier")
    print(f"  Total params:     {total:,}")
    print(f"  Trainable params: {trainable:,}")
    print(f"  Encoder layers:   {model.config.num_hidden_layers}")
    print(f"  Hidden size:      {model.config.hidden_size}")
    print(f"  Pooled dim:       {model.config.hidden_size * 2} (Mean+Max)")
    print(f"  Output classes:   {model.num_labels}")
