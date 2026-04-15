import math
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Utilities
# -----------------------------
def lengths_to_padding_mask(lengths: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
    """
    lengths: [B] int64
    returns padding mask: [B, T] where True indicates PAD (to match PyTorch Transformer convention)
    """
    B = lengths.size(0)
    if max_len is None:
        max_len = int(lengths.max().item())
    arange = torch.arange(max_len, device=lengths.device).unsqueeze(0).expand(B, -1)
    return arange >= lengths.unsqueeze(1)


def downsample_mask_1d(sample_mask: torch.Tensor, stride: int, kernel: int) -> torch.Tensor:
    """
    Convert a sample-level boolean mask [B, T] to a frame-level mask [B, T']
    using max-pooling so any masked sample in the receptive field marks the frame as masked.
    """
    # sample_mask: bool -> float
    x = sample_mask.float().unsqueeze(1)  # [B, 1, T]
    # Use max pool as an "any" over window
    y = F.max_pool1d(x, kernel_size=kernel, stride=stride, padding=0)  # [B,1,T']
    return (y.squeeze(1) > 0.0)  # [B, T']


# -----------------------------
# Conv Feature Encoder
# -----------------------------
class ConvFeatureEncoder(nn.Module):
    """
    A 1D CNN stack similar in spirit to wav2vec2 feature encoder.
    Produces frame-level features [B, T', C].
    """
    def __init__(
        self,
        in_channels: int = 1,
        conv_layers: Tuple[Tuple[int, int, int], ...] = (
            # (out_channels, kernel_size, stride)
            (256, 10, 5),
            (256, 8, 4),
            (256, 4, 2),
            (256, 4, 2),
            (256, 4, 2),
        ),
        dropout: float = 0.0,
    ):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        c_in = in_channels
        for c_out, k, s in conv_layers:
            self.conv_layers.append(nn.Conv1d(c_in, c_out, kernel_size=k, stride=s, bias=False))
            self.layer_norms.append(nn.GroupNorm(1, c_out))  # stable for 1D signals
            c_in = c_out

        # For mask downsampling bookkeeping (approximate receptive mapping)
        self.total_stride = 1
        for _, _, s in conv_layers:
            self.total_stride *= s
        self.first_kernel = conv_layers[0][1]
        self.first_stride = conv_layers[0][2]

        self.out_dim = c_in

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T] or [B, 1, T]
        returns: [B, T', C]
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B,1,T]
        for conv, gn in zip(self.conv_layers, self.layer_norms):
            x = conv(x)
            x = gn(x)
            x = F.gelu(x)
            x = self.dropout(x)
        x = x.transpose(1, 2)  # [B,T',C]
        return x


# -----------------------------
# Transformer Context Network
# -----------------------------
class TransformerContext(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        n_layers: int = 6,
        dim_ff: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.final_ln = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [B,T,C]
        key_padding_mask: [B,T] True for PAD
        """
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        return self.final_ln(x)


# -----------------------------
# HuBERT-style ECG Model
# -----------------------------
@dataclass
class ECGHuBERTOutput:
    logits_masked: torch.Tensor          # [N_masked, K]
    target_masked: Optional[torch.Tensor]  # [N_masked]
    frame_mask: torch.Tensor             # [B, T']
    features: torch.Tensor               # [B, T', C]
    context: torch.Tensor                # [B, T', C]


class ECGHuBERTModel(nn.Module):
    """
    HuBERT-style:
      waveform -> conv features -> mask frames -> transformer -> predict cluster IDs (K)

    Note:
      - Targets (cluster IDs per frame) should come from offline k-means on conv features.
      - This model expects you to provide frame-level targets at training time.
    """
    def __init__(
        self,
        sampling_rate: int = 1000,
        feature_dim: int = 256,
        num_clusters: int = 100,          # K in HuBERT
        mask_emb_init_std: float = 0.02,
        conv_layers: Tuple[Tuple[int, int, int], ...] = (
            (256, 10, 5),
            (256, 8, 4),
            (256, 4, 2),
            (256, 4, 2),
            (256, 4, 2),
        ),
        tf_layers: int = 6,
        tf_heads: int = 8,
        tf_ff: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.sr = sampling_rate
        self.num_clusters = num_clusters

        self.feature_extractor = ConvFeatureEncoder(
            in_channels=1,
            conv_layers=conv_layers,
            dropout=0.0,
        )
        conv_out = self.feature_extractor.out_dim

        # project conv output to transformer dim (feature_dim)
        self.post_proj = nn.Linear(conv_out, feature_dim)
        self.dropout = nn.Dropout(dropout)

        # learned mask embedding applied at feature-frame level
        self.mask_emb = nn.Parameter(torch.empty(feature_dim))
        nn.init.normal_(self.mask_emb, mean=0.0, std=mask_emb_init_std)

        self.context_net = TransformerContext(
            d_model=feature_dim,
            n_heads=tf_heads,
            n_layers=tf_layers,
            dim_ff=tf_ff,
            dropout=dropout,
        )

        # prediction head: context -> cluster logits
        self.pred_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, num_clusters),
        )

    @torch.no_grad()
    def infer_num_frames(self, num_samples: int) -> int:
        """
        Compute output length after conv feature extractor.
        """
        # simulate length propagation (valid conv)
        L = num_samples
        for conv in self.feature_extractor.conv_layers:
            k = conv.kernel_size[0]
            s = conv.stride[0]
            L = 1 + (L - k) // s
        return max(int(L), 0)

    def forward(
        self,
        input_values: torch.Tensor,              # [B, T] float32
        frame_mask: Optional[torch.Tensor] = None,   # [B, T] bool, True indicates to be masked (optional)
        lengths: Optional[torch.Tensor] = None,       # [B] true lengths (optional if no padding)
        frame_targets: Optional[torch.Tensor] = None, # [B, T'] int64 cluster ids per frame
        mask_on_features: bool = True,
    ) -> ECGHuBERTOutput:

        B, T = input_values.shape
        feats = self.feature_extractor(input_values)      # [B, T', Cc]
        feats = self.post_proj(feats)                     # [B, T', C]
        feats = self.dropout(feats)

        Tp = feats.size(1)

        # Build frame_mask (which frames should be masked)
        if frame_mask is None:
            frame_mask = torch.zeros((B, Tp), device=feats.device, dtype=torch.bool)
        else:
            # Approx mapping: use first conv kernel/stride to downsample the mask,
            # then further downsample by total stride ratio with another pooling.
            # Practical and works well; if you want exact mapping, compute per-layer pooling.
            # Here we do a single pooling that roughly matches total stride.
            total_stride = self.feature_extractor.total_stride
            # choose kernel ~ total_stride (conservative)
            pool_k = total_stride
            pool_s = total_stride
            frame_mask = downsample_mask_1d(sample_mask, stride=pool_s, kernel=pool_k)

            # frame_mask length may differ slightly from feats due to conv kernels
            if frame_mask.size(1) != Tp:
                # align by trimming/padding
                if frame_mask.size(1) > Tp:
                    frame_mask = frame_mask[:, :Tp]
                else:
                    pad = Tp - frame_mask.size(1)
                    frame_mask = F.pad(frame_mask, (0, pad), value=False)

        # Apply feature masking (HuBERT-style)
        if mask_on_features and frame_mask.any():
            masked_feats = feats.clone()
            masked_feats[frame_mask] = self.mask_emb
        else:
            masked_feats = feats

        # Padding mask for transformer (True=PAD)
        if lengths is None:
            key_padding_mask = None
        else:
            # convert sample lengths to frame lengths
            frame_lengths = torch.tensor(
                [self.infer_num_frames(int(l.item())) for l in lengths],
                device=lengths.device,
                dtype=torch.long,
            )
            key_padding_mask = lengths_to_padding_mask(frame_lengths, max_len=Tp)

        ctx = self.context_net(masked_feats, key_padding_mask=key_padding_mask)  # [B,T',C]
        logits = self.pred_head(ctx)                                             # [B,T',K]

        # Return logits only for masked positions (HuBERT loss is computed on masked frames)
        if frame_mask.any():
            logits_masked = logits[frame_mask]  # [N_masked, K]
            target_masked = frame_targets[frame_mask] if frame_targets is not None else None
        else:
            logits_masked = logits.reshape(-1, logits.size(-1))
            target_masked = frame_targets.reshape(-1) if frame_targets is not None else None

        return ECGHuBERTOutput(
            logits_masked=logits_masked,
            target_masked=target_masked,
            frame_mask=frame_mask,
            features=feats,
            context=ctx,
        )


# -----------------------------
# Loss helper
# -----------------------------
def hubert_ce_loss(logits_masked: torch.Tensor, target_masked: torch.Tensor) -> torch.Tensor:
    """
    Cross-entropy on masked frames only.
    """
    return F.cross_entropy(logits_masked, target_masked)