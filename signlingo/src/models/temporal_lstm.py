import torch
import torch.nn as nn
from typing import Optional, Tuple


class TemporalLSTM(nn.Module):
    """Bidirectional LSTM for temporal motion trajectory modeling."""

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.directions = 2

        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
            bidirectional=True,
        )

    def _init_hidden(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        layers = self.num_layers * self.directions
        h = torch.zeros(
            layers, batch_size, self.hidden_dim, device=device, dtype=dtype
        )
        c = torch.zeros(
            layers, batch_size, self.hidden_dim, device=device, dtype=dtype
        )
        return h, c

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        x: [B, T, input_dim]
        hidden: optional (h, c). For bidirectional full-sequence passes (typical
            inference over a sliding window), pass None each forward.
        Returns:
            out: [B, T, hidden_dim * directions]
            hidden: (h_n, c_n), detached from the graph
        """
        batch_size = x.size(0)
        if hidden is None:
            hidden = self._init_hidden(batch_size, x.device, x.dtype)
        else:
            hidden = tuple(t.to(device=x.device, dtype=x.dtype) for t in hidden)

        out, hidden = self.lstm(x, hidden)
        hidden = tuple(h.detach() for h in hidden)
        return out, hidden

    def reset_state(self) -> None:
        """No internal state; pipeline manages sequencing. Kept for API compatibility."""
        pass
