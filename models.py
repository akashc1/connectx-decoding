import torch
from torch import nn
from torch.nn import functional as F


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
        target_regions=None,
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
            custom_weights = get_connectome_weights(target_regions)
            self.attn_mask = torch.tensor(custom_weights, requires_grad=False)
        else:
            self.key_proj = nn.Linear(d_in, d_hidden)
            self.query_proj = nn.Linear(d_in, d_hidden)

        self.out_proj = nn.Linear(d_hidden, d_out)

    def forward(self, x, attn_mask):
        B, R, C = x.shape
        val = self.val_proj(x)

        if self.use_custom_weights:
            return self.attn_mask @ (val * attn_mask.byte())

        k = self.key_proj(x).view(B, R, self.num_heads, C // self.num_heads).transpose(1, 2)
        q = self.query_proj(x).view(B, R, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = val.view(B, R, self.num_heads, C // self.num_heads).transpose(1, 2)

        raw_scores = (q @ k.transpose(-2, -1)) / (self.d_hidden ** -0.5)
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
        num_regions=10,
        hidden_dim=64,
        conv_w=7,
        num_convs=3,
        t_in=120,
        squash_neurons='mean',
        num_input_neurons=None,
        flatten_dim=None,
        use_connectome_attn_weights=False,
    ):
        super().__init__()

        self.use_connectome_attn_weights = use_connectome_attn_weights
        self.hidden_dim = hidden_dim

        self.region_embed = nn.Embedding(num_regions, hidden_dim, padding_idx=-1)

        # An initial convolution to preprocess everything
        self.squash_neurons = squash_neurons
        num_inputs = num_input_neurons or 1
        self.initial_conv = nn.Conv1d(num_inputs, hidden_dim, conv_w, padding=conv_w // 2)

        # Some more convs to process information across time
        convs = []
        for _ in num_convs:
            convs.append(nn.Conv1d(hidden_dim, hidden_dim, conv_w, padding=conv_w // 2))
            convs.append(nn.BatchNorm1d(hidden_dim))
            convs.append(nn.ReLU())
        self.convs = nn.Sequential(*convs)

        self.flatten_dim = flatten_dim or hidden_dim
        self.pre_attn_proj = (
            nn.Linear(hidden_dim * t_in, flatten_dim)
            if flatten_dim is not None
            else nn.Identity()
        )

    def forward(self, x):
        """
        x: (B, num_regions, num_neurons, d_time)
        """
        B, num_regions, num_neurons, d_time = x.shape
        assert num_regions == self.region_embed.weight.shape[0]
        x = x.view(-1, num_neurons, d_time)

        for c in (self.initial_conv, F.relu, self.convs):
            x = c(x)

        x = x.view(B, num_regions, -1)
        x = self.pre_attn_proj(x)
