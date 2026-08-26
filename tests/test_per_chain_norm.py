"""PerChainNormEncoder — per-chain LayerNorm on top of a shared backbone.

Verifies the three properties the trainer relies on:
  1. chain routing by global node id is exact at partition boundaries;
  2. the two chains pass through DIFFERENT LayerNorm modules (gradients
     flow to the right one);
  3. a checkpoint round-trip preserves chain_starts (self-describing).
"""

import torch

from openheads.models import PerChainNormEncoder

ETH_N = 7  # toy partition: eth [0..6], tron [7..19]
N_TOTAL = 20


def _toy_encoder() -> PerChainNormEncoder:
    torch.manual_seed(0)
    return PerChainNormEncoder(in_channels=5, hidden_channels=8, chain_starts=[0, ETH_N])


def _toy_graph() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(1)
    x = torch.randn(N_TOTAL, 5)
    src = torch.arange(N_TOTAL)
    dst = (src + 1) % N_TOTAL
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
    return x, edge_index


class TestChainRouting:
    def test_boundary_ids_exact(self) -> None:
        enc = _toy_encoder()
        ids = torch.tensor([0, ETH_N - 1, ETH_N, N_TOTAL - 1])
        assert enc.chain_ids_of(ids).tolist() == [0, 0, 1, 1]

    def test_rejects_bad_chain_starts(self) -> None:
        import pytest

        with pytest.raises(ValueError):
            PerChainNormEncoder(in_channels=5, chain_starts=[3, 7])
        with pytest.raises(ValueError):
            PerChainNormEncoder(in_channels=5, chain_starts=None)


class TestPerChainNorm:
    def test_chains_use_distinct_norms(self) -> None:
        enc = _toy_encoder()
        enc.eval()
        x, edge_index = _toy_graph()
        node_ids = torch.arange(N_TOTAL)
        # Make the two LayerNorms differ, then check the same raw embedding
        # row would be normalised differently depending on its chain.
        with torch.no_grad():
            enc.chain_norms[1].weight.fill_(3.0)
            enc.chain_norms[1].bias.fill_(1.5)
        out = enc.get_embeddings(x, edge_index, node_ids)
        raw = enc.backbone.get_embeddings(x, edge_index)
        eth_expected = enc.chain_norms[0](raw[:ETH_N])
        tron_expected = enc.chain_norms[1](raw[ETH_N:])
        assert torch.allclose(out[:ETH_N], eth_expected, atol=1e-6)
        assert torch.allclose(out[ETH_N:], tron_expected, atol=1e-6)
        assert not torch.allclose(out[ETH_N:], enc.chain_norms[0](raw[ETH_N:]))

    def test_gradients_route_to_own_norm(self) -> None:
        enc = _toy_encoder()
        x, edge_index = _toy_graph()
        eth_only = torch.arange(0, ETH_N)
        out = enc.get_embeddings(x[:ETH_N], edge_index[:, :4] % ETH_N, eth_only)
        out.sum().backward()
        assert enc.chain_norms[0].weight.grad is not None
        assert enc.chain_norms[1].weight.grad is None


class TestCheckpointRoundTrip:
    def test_chain_starts_survive_state_dict(self) -> None:
        enc = _toy_encoder()
        state = enc.state_dict()
        fresh = PerChainNormEncoder(in_channels=5, hidden_channels=8, chain_starts=[0, 999])
        fresh.load_state_dict(state)
        assert fresh.chain_starts.tolist() == [0, ETH_N]
