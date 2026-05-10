import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GoldenGateRoPE2d(nn.Module):
    """
    2D rotary position embeddings applied across heads and per-head feature dims.

    Expects input shaped (N, H, W, h, d), where d is even and split into (x, y) pairs.
    """

    def __init__(
        self,
        image_size: Tuple[int, int],
        n_heads: int,
        head_dim: int,
        min_freq: float,
        max_freq: float,
        p_zero_freqs: float = 0.0,
        direction_spacing: float = math.pi * (math.sqrt(5) - 1) / 2,
    ):
        super().__init__()
        assert head_dim % 2 == 0
        assert 0 <= p_zero_freqs <= 1

        n_freqs = head_dim // 2
        n_zero_freqs = round(p_zero_freqs * n_freqs)

        omega_F = torch.cat(
            (
                torch.zeros(n_zero_freqs),
                min_freq
                * (max_freq / min_freq) ** torch.linspace(0, 1, n_freqs - n_zero_freqs),
            )
        )
        phi_hF = (
            torch.arange(n_heads * n_freqs).reshape(n_heads, n_freqs) * direction_spacing
        )
        directions_hF2 = torch.stack((torch.cos(phi_hF), torch.sin(phi_hF)), dim=-1)
        freqs_hF2 = omega_F.unsqueeze(-1) * directions_hF2

        H, W = image_size
        xlim, ylim = math.sqrt(W / H), math.sqrt(H / W)
        x_HW = torch.linspace(-xlim, xlim, W).reshape(1, W).expand(H, W)
        y_HW = torch.linspace(-ylim, ylim, H).reshape(H, 1).expand(H, W)
        positions_HW112 = torch.stack((x_HW, y_HW), dim=-1).reshape(H, W, 1, 1, 2)

        theta_HWhF = (freqs_hF2 * positions_HW112).sum(dim=-1)
        self.register_buffer("cos_HWhF", torch.cos(theta_HWhF))
        self.register_buffer("sin_HWhF", torch.sin(theta_HWhF))

    def forward(self, input_NHWhd: torch.Tensor) -> torch.Tensor:
        x_NHWhF, y_NHWhF = input_NHWhd.float().chunk(2, dim=-1)
        x_out_NHWhF = x_NHWhF * self.cos_HWhF - y_NHWhF * self.sin_HWhF
        y_out_NHWhF = x_NHWhF * self.sin_HWhF + y_NHWhF * self.cos_HWhF
        output_NHWhd = torch.cat((x_out_NHWhF, y_out_NHWhF), dim=-1)
        return output_NHWhd.type_as(input_NHWhd)


class MultiHeadSelfAttention2D(nn.Module):
    """
    Global MHSA over H*W tokens with 2D RoPE applied to q and k.
    Input/Output tensor shape: (N, H, W, D)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        image_size: Tuple[int, int],
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        rope_min_freq: float = 1.0,
        rope_max_freq: float = 10000.0,
        rope_zero_freq_ratio: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.rope = GoldenGateRoPE2d(
            image_size=image_size,
            n_heads=num_heads,
            head_dim=self.head_dim,
            min_freq=rope_min_freq,
            max_freq=rope_max_freq,
            p_zero_freqs=rope_zero_freq_ratio,
        )

        self.image_size = image_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, H, W, D = x.shape
        assert (H, W) == self.image_size, "Input spatial dims must match configured image_size"

        qkv = self.qkv(x)  # (N, H, W, 3*D)
        q, k, v = qkv.chunk(3, dim=-1)

        def reshape_heads(t: torch.Tensor) -> torch.Tensor:
            t = t.view(N, H, W, self.num_heads, self.head_dim)
            return t

        q = reshape_heads(q)
        k = reshape_heads(k)
        v = reshape_heads(v)

        q = self.rope(q)
        k = self.rope(k)

        # Flatten spatial dims for attention
        q = q.permute(0, 3, 1, 2, 4).contiguous().view(N, self.num_heads, H * W, self.head_dim)
        k = k.permute(0, 3, 1, 2, 4).contiguous().view(N, self.num_heads, H * W, self.head_dim)
        v = v.permute(0, 3, 1, 2, 4).contiguous().view(N, self.num_heads, H * W, self.head_dim)

        scale = self.head_dim ** -0.5
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (N, h, S, S)
        attn = attn_scores.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)  # (N, h, S, d)
        out = out.transpose(1, 2).contiguous().view(N, H, W, self.dim)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class TransformerMLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock2D(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        image_size: Tuple[int, int],
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        drop: float = 0.0,
        qkv_bias: bool = True,
        rope_min_freq: float = 1.0,
        rope_max_freq: float = 10000.0,
        rope_zero_freq_ratio: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention2D(
            dim=dim,
            num_heads=num_heads,
            image_size=image_size,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            rope_min_freq=rope_min_freq,
            rope_max_freq=rope_max_freq,
            rope_zero_freq_ratio=rope_zero_freq_ratio,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = TransformerMLP(dim=dim, mlp_ratio=mlp_ratio, dropout=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, h, w, d = x.shape
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class RecurrentTransformer2D(nn.Module):
    """
    Recurrent transformer operating on a 2D grid. At each recurrent step, it applies
    a stack of transformer blocks to the current latent state and produces an output.

    Input:  (N, C_in, H, W)
    Output: (N, C_out, H, W)
    Hidden state is maintained in (N, H, W, D) format for numerically stable LayerNorm.
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (30, 30),
        in_channels: int = 1,
        out_channels: int = 1,
        dim: int = 256,
        depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        drop: float = 0.0,
        steps: int = 4,
        rope_min_freq: float = 1.0,
        rope_max_freq: float = 10000.0,
        rope_zero_freq_ratio: float = 0.0,
    ):
        super().__init__()

        self.image_size = image_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim = dim
        self.steps = steps

        H, W = image_size
        # 1x1 convolutions for in/out projections
        self.input_proj = nn.Conv2d(in_channels, dim, kernel_size=1)
        self.output_proj = nn.Conv2d(dim, out_channels, kernel_size=1)

        blocks = []
        for _ in range(depth):
            blocks.append(
                TransformerBlock2D(
                    dim=dim,
                    num_heads=num_heads,
                    image_size=image_size,
                    mlp_ratio=mlp_ratio,
                    attn_drop=attn_drop,
                    drop=drop,
                    qkv_bias=True,
                    rope_min_freq=rope_min_freq,
                    rope_max_freq=rope_max_freq,
                    rope_zero_freq_ratio=rope_zero_freq_ratio,
                )
            )
        self.blocks = nn.ModuleList(blocks)

        # Optional gating to inject input signal at every step
        self.input_gate = nn.Parameter(torch.tensor(0.0))

    @torch.no_grad()
    def init_state(self, batch_size: int, device: Optional[torch.device] = None) -> torch.Tensor:
        H, W = self.image_size
        if device is None:
            device = next(self.parameters()).device
        return torch.zeros(batch_size, H, W, self.dim, device=device)

    def forward(
        self,
        x: torch.Tensor,
        steps: Optional[int] = None,
        state: Optional[torch.Tensor] = None,
        return_state: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        assert x.dim() == 4, "x must be NCHW"
        N, C, H, W = x.shape
        assert (H, W) == self.image_size, "Input must be 30x30 (or configured image_size)"

        T = steps if steps is not None else self.steps

        # Project input to latent and convert to NHWD
        x_latent = self.input_proj(x)  # (N, D, H, W)
        x_latent = x_latent.permute(0, 2, 3, 1).contiguous()  # (N, H, W, D)

        if state is None:
            h_state = x_latent
        else:
            h_state = state

        for _ in range(T):
            # Inject input signal via gated residual at each step
            h_state = h_state + torch.tanh(self.input_gate) * x_latent
            for block in self.blocks:
                h_state = block(h_state)

        y = h_state.permute(0, 3, 1, 2).contiguous()  # (N, D, H, W)
        y = self.output_proj(y)  # (N, C_out, H, W)

        if return_state:
            return y, h_state
        return y


__all__ = [
    "GoldenGateRoPE2d",
    "MultiHeadSelfAttention2D",
    "TransformerBlock2D",
    "RecurrentTransformer2D",
]


