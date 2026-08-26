"""Feature engineering pipeline for blockchain graph data.

Reads Parquet transaction files, computes per-address node features, builds
graph structure, and outputs PyG-ready Data objects.

Designed for Ethereum-style (account-based) transaction data.

Features per node (address):
    0. tx_count_out        - outgoing transactions
    1. tx_count_in         - incoming transactions
    2. total_sent          - total value sent (ETH)
    3. total_received      - total value received (ETH)
    4. avg_sent            - average outgoing amount
    5. avg_received        - average incoming amount
    6. max_sent            - largest outgoing transaction
    7. max_received        - largest incoming transaction
    8. unique_receivers    - distinct addresses sent to
    9. unique_senders      - distinct addresses received from
    10. active_days        - distinct days with activity
    11. lifetime_days      - first_seen → last_seen span
    12. in_out_ratio       - in_degree / (in_degree + out_degree)
    13. round_amount_ratio - fraction of tx with round amounts (1, 10, 100, ...)
    14. inter_tx_entropy   - entropy of inter-transaction time intervals
    15. balance_ratio      - (received - sent) / (received + sent + 1e-18)
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

N_FEATURES = 16

WEI_TO_ETH = 1e-18

ROUND_AMOUNTS_WEI = {int(x * 1e18) for x in [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 50, 100, 500, 1000]}


def _entropy(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    total = sum(values)
    if total == 0:
        return 0.0
    probs = [v / total for v in values if v > 0]
    return -sum(p * math.log2(p) for p in probs if p > 0)


def _inter_tx_entropy(timestamps: pd.Series) -> float:
    if len(timestamps) < 2:
        return 0.0
    sorted_ts = timestamps.sort_values()
    diffs = sorted_ts.diff().dropna().dt.total_seconds()
    if len(diffs) == 0:
        return 0.0
    bins = [0, 60, 300, 3600, 86400, 604800, float("inf")]
    hist = Counter()
    for d in diffs:
        for i in range(len(bins) - 1):
            if bins[i] <= d < bins[i + 1]:
                hist[i] += 1
                break
    return _entropy(list(hist.values()))


def _is_round_amount(value: int) -> bool:
    if value == 0:
        return False
    return value in ROUND_AMOUNTS_WEI or (value % int(1e16) == 0)


def load_parquet_transactions(
    paths: list[str | Path],
    value_col: str = "value",
    from_col: str = "from_address",
    to_col: str = "to_address",
    ts_col: str = "block_timestamp",
) -> pd.DataFrame:
    dfs = []
    for p in paths:
        logger.info("loading_parquet", path=str(p))
        df = pd.read_parquet(p, columns=[from_col, to_col, value_col, ts_col])
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    df = df.rename(
        columns={from_col: "from_addr", to_col: "to_addr", value_col: "value", ts_col: "ts"}
    )
    df["value_int"] = df["value"].apply(lambda v: int(v) if pd.notna(v) else 0)
    df["value_eth"] = df["value_int"] * WEI_TO_ETH
    if not pd.api.types.is_datetime64_any_dtype(df["ts"]):
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["date"] = df["ts"].dt.date
    logger.info("loaded_transactions", rows=len(df))
    return df


def compute_node_features(df: pd.DataFrame) -> tuple[dict[str, int], list[list[float]]]:
    all_addrs = set(df["from_addr"].dropna().unique()) | set(df["to_addr"].dropna().unique())
    all_addrs.discard(None)
    addr_to_idx = {addr: i for i, addr in enumerate(sorted(all_addrs))}

    out_stats = df.groupby("from_addr").agg(
        tx_count_out=("value_eth", "count"),
        total_sent=("value_eth", "sum"),
        avg_sent=("value_eth", "mean"),
        max_sent=("value_eth", "max"),
        unique_receivers=("to_addr", "nunique"),
    )

    in_stats = df.groupby("to_addr").agg(
        tx_count_in=("value_eth", "count"),
        total_received=("value_eth", "sum"),
        avg_received=("value_eth", "mean"),
        max_received=("value_eth", "max"),
        unique_senders=("from_addr", "nunique"),
    )

    active_days_out = df.groupby("from_addr")["date"].nunique().rename("active_days_out")
    active_days_in = df.groupby("to_addr")["date"].nunique().rename("active_days_in")

    first_seen_out = df.groupby("from_addr")["ts"].min().rename("first_out")
    last_seen_out = df.groupby("from_addr")["ts"].max().rename("last_out")
    first_seen_in = df.groupby("to_addr")["ts"].min().rename("first_in")
    last_seen_in = df.groupby("to_addr")["ts"].max().rename("last_in")

    round_out = df.copy()
    round_out["is_round"] = round_out["value_int"].apply(_is_round_amount)
    round_ratio_out = round_out.groupby("from_addr")["is_round"].mean().rename("round_ratio_out")

    # Pre-aggregate timestamps per address for entropy (avoid N² scan)
    ts_by_addr: dict[str, list] = {}
    for _, row in df[["from_addr", "ts"]].iterrows():
        a = row["from_addr"]
        if a:
            ts_by_addr.setdefault(a, []).append(row["ts"])
    for _, row in df[["to_addr", "ts"]].iterrows():
        a = row["to_addr"]
        if a:
            ts_by_addr.setdefault(a, []).append(row["ts"])

    inter_entropy = {}
    for addr, timestamps in ts_by_addr.items():
        if len(timestamps) > 1:
            inter_entropy[addr] = _inter_tx_entropy(pd.Series(timestamps))
        else:
            inter_entropy[addr] = 0.0

    features = []
    for addr in sorted(all_addrs):
        tx_out = out_stats.loc[addr, "tx_count_out"] if addr in out_stats.index else 0
        tx_in = in_stats.loc[addr, "tx_count_in"] if addr in in_stats.index else 0
        sent = out_stats.loc[addr, "total_sent"] if addr in out_stats.index else 0
        received = in_stats.loc[addr, "total_received"] if addr in in_stats.index else 0
        avg_s = out_stats.loc[addr, "avg_sent"] if addr in out_stats.index else 0
        avg_r = in_stats.loc[addr, "avg_received"] if addr in in_stats.index else 0
        max_s = out_stats.loc[addr, "max_sent"] if addr in out_stats.index else 0
        max_r = in_stats.loc[addr, "max_received"] if addr in in_stats.index else 0
        uniq_recv = out_stats.loc[addr, "unique_receivers"] if addr in out_stats.index else 0
        uniq_send = in_stats.loc[addr, "unique_senders"] if addr in in_stats.index else 0

        ad_out = active_days_out.get(addr, 0)
        ad_in = active_days_in.get(addr, 0)
        active = max(ad_out, ad_in)

        _ts_max = (
            pd.Timestamp.max.tz_localize("UTC")
            if pd.Timestamp.max.tzinfo is None
            else pd.Timestamp.max
        )
        _ts_min = (
            pd.Timestamp.min.tz_localize("UTC")
            if pd.Timestamp.min.tzinfo is None
            else pd.Timestamp.min
        )
        first = min(
            first_seen_out.get(addr, _ts_max),
            first_seen_in.get(addr, _ts_max),
        )
        last = max(
            last_seen_out.get(addr, _ts_min),
            last_seen_in.get(addr, _ts_min),
        )
        lifetime = max((last - first).days, 0) if first != pd.Timestamp.max else 0

        total_degree = tx_in + tx_out
        in_out = tx_in / total_degree if total_degree > 0 else 0.5

        rr = round_ratio_out.get(addr, 0.0)
        ie = inter_entropy.get(addr, 0.0)
        balance = (received - sent) / (received + sent + 1e-18)

        features.append(
            [
                float(tx_out),
                float(tx_in),
                float(sent),
                float(received),
                float(avg_s),
                float(avg_r),
                float(max_s),
                float(max_r),
                float(uniq_recv),
                float(uniq_send),
                float(active),
                float(lifetime),
                float(in_out),
                float(rr),
                float(ie),
                float(balance),
            ]
        )

    logger.info("computed_features", n_nodes=len(features), n_features=N_FEATURES)
    return addr_to_idx, features


def build_edge_index(
    df: pd.DataFrame,
    addr_to_idx: dict[str, int],
) -> tuple[list[int], list[int]]:
    src = []
    dst = []
    for _, row in df[["from_addr", "to_addr"]].drop_duplicates().iterrows():
        f = row["from_addr"]
        t = row["to_addr"]
        if f in addr_to_idx and t in addr_to_idx:
            src.append(addr_to_idx[f])
            dst.append(addr_to_idx[t])
            src.append(addr_to_idx[t])
            dst.append(addr_to_idx[f])
    logger.info("built_edges", n_edges=len(src))
    return src, dst


def build_pyg_data(
    features: list[list[float]],
    edge_src: list[int],
    edge_dst: list[int],
    labels: dict[str, int] | None = None,
    addr_to_idx: dict[str, int] | None = None,
) -> Any:
    try:
        import torch
        from torch_geometric.data import Data
    except ImportError:
        logger.error("torch_not_available")
        return None

    x = torch.tensor(features, dtype=torch.float)

    x_min = x.min(dim=0).values
    x_max = x.max(dim=0).values
    x_range = x_max - x_min
    x_range[x_range < 1e-8] = 1.0
    x = (x - x_min) / x_range

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)

    y = None
    if labels and addr_to_idx:
        y_list = [-1] * len(features)
        for addr, label in labels.items():
            if addr in addr_to_idx:
                y_list[addr_to_idx[addr]] = label
        y = torch.tensor(y_list, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=y)
    data.n_features = N_FEATURES
    logger.info("built_pyg_data", nodes=x.shape[0], edges=edge_index.shape[1], features=x.shape[1])
    return data
