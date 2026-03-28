import torch
import torch.nn as nn
from typing import Optional


class TemporalLSTM(nn.Module):
    """Bidirectional LSTM for temporal motion trajectory modeling."""

    def __init__(self, input_dim: int = 1024, hidden_dim: int = 256, num_layers: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        # Output dim: hidden_dim * 2 = 512
        self._hidden: Optional[tuple] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, 1024]
        returns: [B, 512] — last hidden state from both directions concatenated
        """
        out, (h_n, _) = self.lstm(x, self._hidden)
        self._hidden = (h_n.detach(), _[0].detach() if isinstance(_, torch.Tensor) else _.detach())

        # h_n: [num_layers * 2, B, hidden_dim]
        # Take last layer's forward and backward hidden states
        forward_h = h_n[-2]   # [B, hidden_dim]
        backward_h = h_n[-1]  # [B, hidden_dim]
        return torch.cat([forward_h, backward_h], dim=-1)  # [B, 512]

    def reset_state(self):
        """Clear LSTM hidden state between signs."""
        self._hidden = None
