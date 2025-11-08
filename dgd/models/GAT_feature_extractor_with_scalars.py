from __future__ import annotations

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as geom_nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class GAT(BaseFeaturesExtractor):
    """
    Pre-norm GATv2 stack with residual projections, optional node-feature
    embedding and attention pooling. (Now concatenates 2 normalized scalars.)
    """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 128, *, embed_dim: 'int | None' = None):
        super().__init__(observation_space, features_dim)

        in_node_dim = observation_space["node_features"].shape[1]  # =4 (one-hot)
        self.node_embed: 'nn.Module | None' = None
        if embed_dim is not None:
            # one-hot (N,4) · W(4,embed_dim)  →  (N,embed_dim)
            self.node_embed = nn.Linear(in_node_dim, embed_dim, bias=False)
            in_node_dim = embed_dim                      # GNN now sees embed_dim

        layer_cfg = [(32, 4), (64, 4), (64, 4)]          # (out_dim, heads)
        self.gnn_layers = nn.ModuleList()
        cur_dim = in_node_dim

        for out_dim, heads in layer_cfg:
            out_channels = out_dim * heads               # concat=True
            conv = geom_nn.GATv2Conv(
                in_channels=cur_dim,
                out_channels=out_dim,
                heads=heads,
                concat=True,
                dropout=0.1,
            )
            norm = nn.LayerNorm(cur_dim)                 # pre-norm
            proj = (
                None
                if cur_dim == out_channels
                else nn.Linear(cur_dim, out_channels, bias=False)
            )

            self.gnn_layers.append(nn.ModuleDict({"conv": conv, "norm": norm, "proj": proj}))
            cur_dim = out_channels

        self.global_pool = geom_nn.AttentionalAggregation(
            gate_nn=nn.Sequential(
                nn.Linear(cur_dim, cur_dim // 2),
                nn.ReLU(),
                nn.Linear(cur_dim // 2, 1),
            )
        )
        self.gcn_out_dim = cur_dim

        # --- NEW: capture max_steps from the observation space for normalization ---
        steps_space = observation_space["steps_left"]
        # Box high might be an array; take the scalar
        max_steps_val = float(steps_space.high[0]) if hasattr(steps_space.high, "__len__") else float(steps_space.high)
        # keep as a buffer so it moves with .to(device)
        self.register_buffer("_max_steps_tensor", th.tensor(max_steps_val), persistent=False)

        # First Linear now expects graph embedding + 2 scalars
        hidden_dim = 256
        self.combined_net = nn.Sequential(
            nn.Linear(self.gcn_out_dim + 2, hidden_dim), nn.ReLU(), nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, features_dim)
        )

    def adj_matrix_to_edge_index(self, adj_matrix: th.Tensor) -> th.Tensor:
        row, col = (adj_matrix > 0).nonzero(as_tuple=True)
        edge_index = th.stack([row, col], dim=0).to(adj_matrix.device)
        return edge_index

    def encode_graph(self, x: th.Tensor, adj: th.Tensor) -> th.Tensor:
        if self.node_embed is not None:                  # optional embedding
            x = self.node_embed(x)

        edge_index = self.adj_matrix_to_edge_index(adj)

        for layer in self.gnn_layers:
            h = layer["conv"](layer["norm"](x), edge_index)   # pre-norm
            res = x if layer["proj"] is None else layer["proj"](x)
            x = F.relu(res + h)                               # residual + act

        batch = th.zeros(x.size(0), dtype=th.long, device=x.device)
        return self.global_pool(x, batch)                     # (1, F)

    def process_graph(self, node_features: th.Tensor, adj_matrix: th.Tensor) -> th.Tensor:
        if adj_matrix.dim() == 3:  # batched
            outs = [
                self.encode_graph(node_features[i], adj_matrix[i])   # (1, F)
                for i in range(adj_matrix.size(0))
            ]
            return th.stack(outs)                                     # (B, 1, F)
        return self.encode_graph(node_features, adj_matrix)           # (1, F)

    def forward(self, observation):
        device = next(self.parameters()).device

        node_features = th.as_tensor(observation["node_features"], dtype=th.float32, device=device)
        adj_matrix    = th.as_tensor(observation["adj_matrix"],    dtype=th.float32, device=device)

        graph_emb = self.process_graph(node_features, adj_matrix)   # (B,1,F) or (1,F)
        if graph_emb.dim() == 3:                                    # (B,1,F) -> (B,F)
            graph_emb = graph_emb.squeeze(1)
        if graph_emb.dim() == 2 and graph_emb.size(0) == 1:         # (1,F) -> (1,F), ok; keep 2D for concat
            pass
        if graph_emb.dim() == 1:                                    # fallback: (F,) -> (1,F)
            graph_emb = graph_emb.unsqueeze(0)

        # --- NEW: scalars -> normalized 2D (B,2) ---
        be = th.as_tensor(observation["best_energy"], dtype=th.float32, device=device)   # (B,1) or (1,)
        sl = th.as_tensor(observation["steps_left"],  dtype=th.float32, device=device)   # (B,1) or (1,)

        if be.dim() == 2 and be.size(-1) == 1: be = be.squeeze(-1)   # (B,)
        if sl.dim() == 2 and sl.size(-1) == 1: sl = sl.squeeze(-1)   # (B,)
        if be.dim() == 0: be = be.unsqueeze(0)
        if sl.dim() == 0: sl = sl.unsqueeze(0)

        be = be.clamp_min(1.0)
        sl = sl.clamp(0.0, float(self._max_steps_tensor.item()))

        best_energy_inv = 1.0 / be                                   # (0,1]
        steps_left_frac = sl / self._max_steps_tensor.clamp_min(1)   # [0,1]
        scalar_feats = th.stack([best_energy_inv, steps_left_frac], dim=-1)  # (B,2)

        # Ensure batch dims match
        if graph_emb.size(0) != scalar_feats.size(0):
            # common case: graph_emb is (1,F) while scalar_feats is (B,2) with B>1 (shouldn't happen with SB3)
            # we broadcast the 1 to B just in case
            graph_emb = graph_emb.expand(scalar_feats.size(0), -1)

        fused = th.cat([graph_emb, scalar_feats], dim=-1)            # (B, F+2)
        return self.combined_net(fused)


