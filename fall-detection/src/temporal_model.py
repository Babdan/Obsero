"""
Temporal Model — TCN / GRU / LSTM classifier for fall detection.

The primary architecture is a Temporal Convolutional Network (TCN) with
dilated causal convolutions, residual connections, and weight normalization.
GRU and LSTM are provided as alternatives.

Input:  (batch, time_steps, input_size)
        Runtime config uses 102 features: position + velocity + acceleration.
Output: (batch, 1) — fall probability logit; apply sigmoid for probability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════════
#  TCN Building Blocks
# ═══════════════════════════════════════════════════════════════════════════════

class CausalConv1d(nn.Module):
    """1D causal convolution with left-padding to preserve temporal causality."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
    ):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = weight_norm(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                dilation=dilation,
                padding=self.padding,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        # Remove the extra right-side padding to maintain causal property
        if self.padding > 0:
            out = out[:, :, : -self.padding]
        return out


class TCNBlock(nn.Module):
    """
    Residual TCN block with two causal convolutions, ReLU, and dropout.

    Architecture:
        input → CausalConv → ReLU → Dropout → CausalConv → ReLU → Dropout → (+residual) → ReLU
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.conv1 = CausalConv1d(
            in_channels, out_channels, kernel_size, dilation
        )
        self.conv2 = CausalConv1d(
            out_channels, out_channels, kernel_size, dilation
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # 1×1 convolution for residual connection if channel sizes differ
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.relu(out)
        out = self.dropout(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        return self.relu(out + residual)


class TCN(nn.Module):
    """
    Temporal Convolutional Network with exponentially increasing dilation.

    Dilation schedule: 1, 2, 4, 8, ... for each channel level.
    This ensures the receptive field grows exponentially with depth.
    """

    def __init__(
        self,
        input_size: int,
        channels: list[int],
        kernel_size: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        layers = []
        num_levels = len(channels)

        for i in range(num_levels):
            in_ch = input_size if i == 0 else channels[i - 1]
            out_ch = channels[i]
            dilation = 2**i

            layers.append(
                TCNBlock(in_ch, out_ch, kernel_size, dilation, dropout)
            )

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, time_steps)

        Returns:
            (batch, channels[-1], time_steps)
        """
        return self.network(x)


# ═══════════════════════════════════════════════════════════════════════════════
#  Full Model
# ═══════════════════════════════════════════════════════════════════════════════

class FallDetectionModel(nn.Module):
    """
    Fall detection temporal classifier.

    Supports TCN (default), GRU, and LSTM architectures via config.

    Args:
        config: Dictionary with model hyperparameters.
    """

    def __init__(self, config: dict):
        super().__init__()

        self.architecture = config.get("architecture", "tcn")
        self.input_size = config.get("input_size", 34)

        if self.architecture == "tcn":
            channels = config.get("tcn_channels", [64, 128, 128])
            kernel_size = config.get("tcn_kernel_size", 3)
            dropout = config.get("dropout", 0.3)

            self.tcn = TCN(
                input_size=self.input_size,
                channels=channels,
                kernel_size=kernel_size,
                dropout=dropout,
            )

            final_ch = channels[-1]
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),   # (B, C, T) → (B, C, 1)
                nn.Flatten(),              # (B, C)
                nn.Linear(final_ch, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1),
            )

        elif self.architecture in ("gru", "lstm"):
            hidden_size = config.get("hidden_size", 128)
            num_layers = config.get("num_layers", 2)
            dropout = config.get("dropout", 0.3)
            bidirectional = config.get("bidirectional", False)

            rnn_cls = nn.GRU if self.architecture == "gru" else nn.LSTM
            self.rnn = rnn_cls(
                input_size=self.input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
                bidirectional=bidirectional,
            )

            fc_input = hidden_size * (2 if bidirectional else 1)
            self.classifier = nn.Sequential(
                nn.Linear(fc_input, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1),
            )

        else:
            raise ValueError(f"Unknown architecture: {self.architecture}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (batch, time_steps, input_size) — keypoint sequences.

        Returns:
            (batch, 1) — fall probability (pre-sigmoid logit).
            Apply sigmoid for probability.
        """
        if self.architecture == "tcn":
            # TCN expects (batch, channels, time)
            x = x.permute(0, 2, 1)  # (B, T, C) → (B, C, T)
            features = self.tcn(x)   # (B, C_out, T)
            logits = self.classifier(features)  # (B, 1)

        else:
            # GRU / LSTM: (B, T, C) → hidden states
            output, _ = self.rnn(x)
            # Use last time step's hidden state
            last_hidden = output[:, -1, :]  # (B, hidden)
            logits = self.classifier(last_hidden)  # (B, 1)

        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return sigmoid probabilities instead of raw logits."""
        logits = self.forward(x)
        return torch.sigmoid(logits)


# ═══════════════════════════════════════════════════════════════════════════════
#  Focal Loss for class imbalance
# ═══════════════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance (many non-fall, few fall samples).

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Weight for the positive (fall) class.
        gamma: Focusing parameter. Higher gamma down-weights easy examples more.
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits: (B, 1) raw model output (before sigmoid).
            targets: (B, 1) binary labels (0 or 1).
        """
        probs = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )

        # p_t = p if y=1, else 1-p
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # alpha_t
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # focal weight
        focal_weight = alpha_t * (1 - p_t) ** self.gamma

        return (focal_weight * bce).mean()


# ═══════════════════════════════════════════════════════════════════════════════
#  Model loading utility
# ═══════════════════════════════════════════════════════════════════════════════

def load_model(
    model_path: str,
    config: dict,
    device: Optional[torch.device] = None,
) -> FallDetectionModel:
    """
    Load a trained model from a checkpoint file.

    Args:
        model_path: Path to the .pth weights file.
        config: Model configuration dictionary.
        device: Target device (cpu or cuda). Auto-detected if None.

    Returns:
        Loaded FallDetectionModel in eval mode.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FallDetectionModel(config)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model
