import torch
from torch import nn
from torch.nn import functional as F

from util.data import get_connectome_weights


class MLP(nn.Module):
    def __init__(self, n_layers, inc, hiddenc, outc):
        super().__init__()
        self.initial_proj = nn.Linear(inc, hiddenc)

        layers = []
        for _ in range(n_layers):
            layers.append(nn.Linear(hiddenc, hiddenc))
            layers.append(nn.GELU())

        self.mlp = nn.Sequential(*layers)
        self.out_proj = nn.Linear(hiddenc, outc)

    def forward(self, x):
        return self.out_proj(self.mlp(F.relu(self.initial_proj(x))))


class RegionAttention(nn.Module):
    """
    Compute attention over region tokens. Optionally accept hardcoded attention weights
    """

    def __init__(
        self,
        d_in=64,
        d_hidden=128,
        d_out=64,
        num_attn_heads=4,
        use_custom_weights=False,
    ):
        super().__init__()

        assert (
            d_hidden % num_attn_heads == 0
        ), f"Dimensionality {d_hidden} not compatible with {num_attn_heads=}!"
        self.use_custom_weights = use_custom_weights
        self.num_heads = num_attn_heads
        self.d_hidden = d_hidden

        self.val_proj = nn.Linear(d_in, d_hidden)

        if use_custom_weights:
            custom_weights = get_connectome_weights()
            self.attn_mask = torch.tensor(custom_weights, requires_grad=False)
        else:
            self.key_proj = nn.Linear(d_in, d_hidden)
            self.query_proj = nn.Linear(d_in, d_hidden)

        self.out_proj = nn.Linear(d_hidden, d_out)

    def forward(self, x, attn_mask=None):
        B, R, C = x.shape
        val = self.val_proj(x)

        if self.use_custom_weights:
            val = val * attn_mask.byte() if attn_mask is not None else val
            return self.out_proj(self.attn_mask @ val)

        k = self.key_proj(x).view(B, R, self.num_heads, self.d_hidden // self.num_heads).transpose(1, 2)
        q = self.query_proj(x).view(B, R, self.num_heads, self.d_hidden // self.num_heads).transpose(1, 2)
        v = val.view(B, R, self.num_heads, self.d_hidden // self.num_heads).transpose(1, 2)

        raw_scores = (q @ k.transpose(-2, -1)) / (self.d_hidden ** -0.5)
        if attn_mask is not None:
            raw_scores = raw_scores.masked_fill(attn_mask == 0, float('-inf'))
        attn_scores = F.softmax(raw_scores, -1)
        out = (attn_scores @ v).transpose(1, 2).contiguous().view(B, R, self.d_hidden)

        return self.out_proj(out)


class MovementPredictor(nn.Module):
    """
    Main class to take in a series of neural firings split by brain region
    and predict the behavior (spinning the wheen left/right)
    """

    def __init__(
        self,
        num_regions=7,
        hidden_dim=256,
        conv_w=25,
        num_convs=12,
        t_in=120,
        num_input_neurons=None,
        attn_hidden_dim=256,
        use_connectome_attn_weights=False,
    ):
        super().__init__()

        self.use_connectome_attn_weights = use_connectome_attn_weights
        self.hidden_dim = hidden_dim

        self.region_embed = nn.Embedding(num_regions, hidden_dim)

        # An initial convolution to preprocess everything
        num_inputs = num_input_neurons or 1
        self.initial_conv = nn.Conv1d(num_inputs, hidden_dim, conv_w, padding=conv_w // 2)

        # Some more convs to process information across time
        convs = []
        for _ in range(num_convs):
            convs.append(nn.Conv1d(hidden_dim, hidden_dim, conv_w, padding=conv_w // 2))
            # convs.append(nn.BatchNorm1d(hidden_dim))
            convs.append(nn.ReLU())
        self.convs = nn.Sequential(*convs)

        self.pre_attn_proj = nn.Linear(hidden_dim * t_in, hidden_dim)

        self.attn = RegionAttention(
            hidden_dim, attn_hidden_dim, hidden_dim,
            use_custom_weights=use_connectome_attn_weights,
        )
        self.mlp = MLP(3, hidden_dim, hidden_dim, hidden_dim // 2)
        self.final_proj = nn.Linear(num_regions * (hidden_dim // 2), 1)

    def forward(self, x):
        """
        x: (B, num_regions, num_neurons, d_time)
        """
        B, num_regions, d_time = x.shape
        assert num_regions == self.region_embed.weight.shape[0]
        x = x.view(-1, 1, d_time).float()

        for c in (self.initial_conv, F.relu, self.convs):
            x = c(x)

        x = x.view(B, num_regions, -1)
        x = F.relu(self.pre_attn_proj(x))
        x = x + self.region_embed(torch.arange(num_regions).to(x.device))[None, ...]
        x = F.relu(self.attn(x))
        x = F.gelu(self.mlp(x))
        return self.final_proj(x.flatten(-2))
