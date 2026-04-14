from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GCNConv
from torch_geometric.typing import OptTensor


class Encoder(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation,
        base_model=GCNConv,
        num_layers: int = 2,
    ):
        super().__init__()
        assert num_layers >= 1

        self.base_model = base_model
        self.num_layers = num_layers
        self.activation = activation

        # First layer: project to 2*out_channels unless it's the only layer
        conv_layers = [
            base_model(
                in_channels, out_channels if num_layers == 1 else 2 * out_channels
            )
        ]
        for _ in range(1, num_layers - 1):
            conv_layers.append(base_model(2 * out_channels, 2 * out_channels))
        if num_layers > 1:
            conv_layers.append(base_model(2 * out_channels, out_channels))

        self.conv_layers = nn.ModuleList(conv_layers)

    def forward(self, node_features: Tensor, edge_index: Tensor) -> Tensor:
        node_emb = node_features
        for conv in self.conv_layers:
            node_emb = self.activation(conv(node_emb, edge_index))
        return node_emb


# Shared projection + similarity mixin
class _ProjectionMixin(torch.nn.Module):
    """Shared projection head and cosine-similarity helpers for MV/SV models."""

    def __init__(self, num_hidden: int, num_proj_hidden: int, tau: float):
        super().__init__()
        self.tau = tau
        self.forward_proj_1 = nn.Linear(num_hidden, num_proj_hidden)
        self.forward_proj_2 = nn.Linear(num_proj_hidden, num_hidden)

    def projection(self, gnn_embedding: Tensor) -> Tensor:
        projected = F.elu(self.forward_proj_1(gnn_embedding))
        return self.forward_proj_2(projected)

    def similarity_matrix(self, emb_a: Tensor, emb_b: Tensor) -> Tensor:
        """Normalised dot-product similarity matrix between two embedding sets."""
        emb_a = F.normalize(emb_a)
        emb_b = F.normalize(emb_b)
        return torch.mm(emb_a, emb_b.t())

    def tau_scaling(self, sim: Tensor) -> Tensor:
        """Scales a similarity matrix by temperature tau (used in NT-Xent losses)."""
        return torch.exp(sim / self.tau)

    def _reduce(self, per_node_loss: Tensor, mean: bool) -> Tensor:
        return per_node_loss.mean() if mean else per_node_loss.sum()

    @staticmethod
    def _strip_self_loops(adj: Tensor) -> Tensor:
        """Returns a binarised adjacency matrix with the diagonal zeroed out."""
        adj = adj - torch.diag_embed(adj.diag())
        adj[adj > 0] = 1
        return adj

    @staticmethod
    def _positive_pair_counts(adj: Tensor) -> Tensor:
        """
        Number of positive pairs per node:
          intra-view neighbours + inter-view neighbours + self inter-view = 2*|N_i| + 1
        """
        return torch.squeeze(torch.tensor(torch.sum(adj, 1) * 2 + 1))


# Multi-View model
class MVmodel(_ProjectionMixin):
    def __init__(
        self,
        encoder: Encoder,
        num_hidden: int,
        num_proj_hidden: int,
        tau: float = 0.5,
    ):
        super().__init__(num_hidden, num_proj_hidden, tau)
        self.encoder = encoder

    def forward(self, node_features: Tensor, edge_index: Tensor) -> Tensor:
        gnn_embedding = self.encoder(node_features, edge_index)
        projected_embedding = self.projection(gnn_embedding)
        return projected_embedding

    # Basic (non-neighbour-aware) contrastive loss
    def _semi_loss(self, emb_a: Tensor, emb_b: Tensor) -> Tensor:
        self_sim = self.tau_scaling(self.similarity_matrix(emb_a, emb_a))
        cross_sim = self.tau_scaling(self.similarity_matrix(emb_a, emb_b))
        return -torch.log(
            cross_sim.diag() / (self_sim.sum(1) + cross_sim.sum(1) - self_sim.diag())
        )

    def loss(
        self,
        proj_emb_1: Tensor,
        proj_emb_2: Tensor,
        mean: bool = True,
        batch_size: int = 0,
    ) -> Tensor:
        per_node = (
            self._semi_loss(proj_emb_1, proj_emb_2)
            + self._semi_loss(proj_emb_2, proj_emb_1)
        ) * 0.5
        return self._reduce(per_node, mean)

    # Neighbour-aware contrastive loss

    def _neighbor_contrastive_loss(
        self, emb_a: Tensor, emb_b: Tensor, adj: Tensor
    ) -> Tensor:
        """Neighbour contrastive loss (unbiased variant)."""
        adj = self._strip_self_loops(adj)
        positive_pair_counts = self._positive_pair_counts(adj)

        intra_sim = self.tau_scaling(self.similarity_matrix(emb_a, emb_a))
        inter_sim = self.tau_scaling(self.similarity_matrix(emb_a, emb_b))

        numerator = (
            inter_sim.diag() + intra_sim.mul(adj).sum(1) + inter_sim.mul(adj).sum(1)
        )
        denominator = intra_sim.sum(1) + inter_sim.sum(1) - intra_sim.diag()

        per_node_loss = (numerator / denominator) / positive_pair_counts
        return -torch.log(per_node_loss)

    def contrastive_loss(
        self, emb_a: Tensor, emb_b: Tensor, adj: Tensor, mean: bool = True
    ) -> Tensor:
        per_node = (
            self._neighbor_contrastive_loss(emb_a, emb_b, adj)
            + self._neighbor_contrastive_loss(emb_b, emb_a, adj)
        ) * 0.5
        return self._reduce(per_node, mean)

    def _neighbor_contrastive_loss_biased(
        self, emb_a: Tensor, emb_b: Tensor, adj: Tensor, pseudo_labels: Tensor
    ) -> Tensor:
        """Neighbour contrastive loss with pseudo-label negative masking."""
        adj = self._strip_self_loops(adj)
        positive_pair_counts = self._positive_pair_counts(adj)

        intra_sim = self.tau_scaling(self.similarity_matrix(emb_a, emb_a))
        inter_sim = self.tau_scaling(self.similarity_matrix(emb_a, emb_b))

        # Mask pairs that share the same pseudo-label out of the denominator
        negative_mask = (pseudo_labels.view(-1, 1) != pseudo_labels.view(1, -1)).float()
        masked_intra_sim = intra_sim * negative_mask
        masked_inter_sim = inter_sim * negative_mask

        numerator = (
            inter_sim.diag() + intra_sim.mul(adj).sum(1) + inter_sim.mul(adj).sum(1)
        )
        denominator = (
            masked_intra_sim.sum(1) + masked_inter_sim.sum(1) - intra_sim.diag()
        )

        per_node_loss = (numerator / denominator) / positive_pair_counts
        return -torch.log(per_node_loss)

    def contrastive_loss_biased(
        self,
        emb_a: Tensor,
        emb_b: Tensor,
        adj: Tensor,
        pseudo_labels: Tensor,
        mean: bool = True,
    ) -> Tensor:
        per_node = (
            self._neighbor_contrastive_loss_biased(emb_a, emb_b, adj, pseudo_labels)
            + self._neighbor_contrastive_loss_biased(emb_b, emb_a, adj, pseudo_labels)
        ) * 0.5
        return self._reduce(per_node, mean)


# Single-View model
class SVmodel(_ProjectionMixin):
    def __init__(
        self,
        encoder: Encoder,
        num_hidden: int,
        num_proj_hidden: int,
        tau: float = 0.5,
    ):
        super().__init__(num_hidden, num_proj_hidden, tau)
        self.encoder = encoder

    def forward(self, node_features: Tensor, edge_index: Tensor) -> Tensor:
        gnn_embedding = self.encoder(node_features, edge_index)
        projected_embedding = self.projection(gnn_embedding)
        return projected_embedding

    def _neighbor_contrastive_loss(
        self,
        emb_a: Tensor,
        emb_b: Tensor,
        adj: Tensor,
        sample_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Neighbour contrastive loss with optional per-sample mask."""
        adj = self._strip_self_loops(adj)
        positive_pair_counts = self._positive_pair_counts(adj)

        intra_sim = self.tau_scaling(self.similarity_matrix(emb_a, emb_a))
        inter_sim = self.tau_scaling(self.similarity_matrix(emb_a, emb_b))

        if sample_mask is not None:
            intra_sim = intra_sim * sample_mask
            inter_sim = inter_sim * sample_mask

        numerator = (
            inter_sim.diag() + intra_sim.mul(adj).sum(1) + inter_sim.mul(adj).sum(1)
        )
        denominator = intra_sim.sum(1) + inter_sim.sum(1) - intra_sim.diag()

        per_node_loss = (numerator / denominator) / positive_pair_counts
        return -torch.log(per_node_loss)

    def contrastive_loss(
        self,
        emb_a: Tensor,
        emb_b: Tensor,
        adj: Tensor,
        sample_mask: Optional[Tensor] = None,
        mean: bool = True,
    ) -> Tensor:
        per_node = (
            self._neighbor_contrastive_loss(emb_a, emb_b, adj, sample_mask)
            + self._neighbor_contrastive_loss(emb_b, emb_a, adj, sample_mask)
        ) * 0.5
        return self._reduce(per_node, mean)


# Graph augmentation utilities
def drop_feature(node_features: Tensor, drop_prob: float) -> Tensor:
    """Randomly zeros out feature dimensions with probability `drop_prob`."""
    drop_mask = (
        torch.empty(node_features.size(1), device=torch.device("cpu")).uniform_(0, 1)
        < drop_prob
    )
    node_features = node_features.clone()
    node_features[:, drop_mask] = 0
    return node_features


def filter_adj(
    row: Tensor, col: Tensor, edge_attr: OptTensor, keep_mask: Tensor
) -> Tuple[Tensor, Tensor, OptTensor]:
    """Filters edge endpoints and attributes by a boolean keep mask."""
    filtered_attr = None if edge_attr is None else edge_attr[keep_mask]
    return row[keep_mask], col[keep_mask], filtered_attr


def dropout_adj(
    edge_index: Tensor,
    edge_attr: Tensor,
    force_undirected: bool = False,
    num_nodes: Optional[int] = None,
    training: bool = True,
) -> Tuple[Tensor, Tensor]:
    """Drops edges stochastically using per-edge weights stored in `edge_attr`."""
    if not training:
        return edge_index, edge_attr

    row, col = edge_index

    if force_undirected:
        upper_tri_mask = row <= col
        row, col, edge_attr = (
            row[upper_tri_mask],
            col[upper_tri_mask],
            edge_attr[upper_tri_mask],
        )

    # Each edge is kept if a uniform sample exceeds its weight
    keep_mask = torch.rand(
        edge_attr.size(0), device=torch.device("cpu")
    ) >= edge_attr.to("cpu")
    row, col, edge_attr = filter_adj(row, col, edge_attr, keep_mask)

    if force_undirected:
        edge_index = torch.stack([torch.cat([row, col]), torch.cat([col, row])], dim=0)
    else:
        edge_index = torch.stack([row, col], dim=0)

    return edge_index, edge_attr


def multiple_dropout_average(
    edge_index: Tensor,
    edge_attr: Tensor,
    num_trials: int = 10,
    force_undirected: bool = False,
    num_nodes: Optional[int] = None,
    threshold_ratio: float = 0.5,
    training: bool = True,
    device: str = "cuda",
) -> Tuple[Tensor, Tensor]:
    """
    Optionally runs multiple dropout trials and keeps edges that survive in at
    least `threshold_ratio` of trials (simulation path currently disabled).
    """
    if not training:
        return edge_index, edge_attr

    if num_nodes is None:
        num_nodes = int(edge_index.max().item()) + 1

    edge_index = edge_index.to(device)
    edge_attr = edge_attr.to(device)

    # Simulation path (currently disabled via flag)
    use_simulation = False
    if use_simulation:
        edge_count = torch.zeros(
            (num_nodes, num_nodes), dtype=torch.int32, device=device
        )
        for _ in range(num_trials):
            dropped_edge_index, _ = dropout_adj(edge_index, edge_attr, force_undirected)
            dropped_edge_index = dropped_edge_index.to(device)
            src, dst = dropped_edge_index
            edge_count[src, dst] += 1
            if force_undirected:
                edge_count[dst, src] += 1
        threshold = int(num_trials * threshold_ratio)
        final_edge_index = (edge_count >= threshold).nonzero().t().contiguous()
    else:
        final_edge_index, _ = dropout_adj(edge_index, edge_attr, force_undirected)

    return final_edge_index, edge_attr


def random_dropout_adj(
    edge_index: Tensor,
    edge_attr: OptTensor = None,
    p: float = 0.5,
    force_undirected: bool = False,
    num_nodes: Optional[int] = None,
    training: bool = True,
) -> Tuple[Tensor, OptTensor]:
    r"""Randomly drops edges from the adjacency matrix
    :obj:`(edge_index, edge_attr)` with probability :obj:`p` using samples from
    a Bernoulli distribution.

    .. warning::

        :class:`~torch_geometric.utils.dropout_adj` is deprecated and will
        be removed in a future release.
        Use :class:`torch_geometric.utils.dropout_edge` instead.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)
        p (float, optional): Dropout probability. (default: :obj:`0.5`)
        force_undirected (bool, optional): If set to :obj:`True`, will either
            drop or keep both edges of an undirected edge.
            (default: :obj:`False`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        training (bool, optional): If set to :obj:`False`, this operation is a
            no-op. (default: :obj:`True`)

    Examples:

        >>> edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
        ...                            [1, 0, 2, 1, 3, 2]])
        >>> edge_attr = torch.tensor([1, 2, 3, 4, 5, 6])
        >>> random_dropout_adj(edge_index, edge_attr)
        (tensor([[0, 1, 2, 3],
                [1, 2, 3, 2]]),
        tensor([1, 3, 5, 6]))

        >>> # The returned graph is kept undirected
        >>> random_dropout_adj(edge_index, edge_attr, force_undirected=True)
        (tensor([[0, 1, 2, 1, 2, 3],
                [1, 2, 3, 0, 1, 2]]),
        tensor([1, 3, 5, 1, 3, 5]))
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"Dropout probability has to be between 0 and 1 (got {p})")

    if not training or p == 0.0:
        return edge_index, edge_attr

    row, col = edge_index
    keep_mask = torch.rand(row.size(0), device=torch.device("cpu")) >= p

    if force_undirected:
        keep_mask[row > col] = False

    row, col, edge_attr = filter_adj(row, col, edge_attr, keep_mask)

    if force_undirected:
        edge_index = torch.stack([torch.cat([row, col]), torch.cat([col, row])], dim=0)
        if edge_attr is not None:
            edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
    else:
        edge_index = torch.stack([row, col], dim=0)

    return edge_index, edge_attr


# ---------------------------------------------------------------------------
# Discriminator
# ---------------------------------------------------------------------------


class Discriminator(nn.Module):
    _NUM_LAYERS = 1
    _HIDDEN_DIM = 64
    _DROPOUT = 0.2
    _INPUT_DROPOUT = 0.1

    def __init__(self, input_dim: int):
        super().__init__()

        layers: list[nn.Module] = [nn.Dropout(self._INPUT_DROPOUT)]
        for i in range(self._NUM_LAYERS + 1):
            layer_in = input_dim if i == 0 else self._HIDDEN_DIM
            layer_out = 1 if i == self._NUM_LAYERS else self._HIDDEN_DIM
            layers.append(nn.Linear(layer_in, layer_out))
            if i < self._NUM_LAYERS:
                layers += [nn.ReLU(), nn.Dropout(self._DROPOUT)]

        self.layers = nn.Sequential(*layers)

    def forward(self, node_features: Tensor) -> Tensor:
        return self.layers(node_features).view(-1)
