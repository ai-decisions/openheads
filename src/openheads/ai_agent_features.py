"""AI-agent behavioral feature engineering for Ethereum addresses.

Computes features designed to distinguish autonomous AI agents from human users
based on behavioral signatures in on-chain activity.

Features are additive to the existing 16-feature pipeline in feature_pipeline.py.
Combined 16 + 20 = 36-dim feature vector per address.
"""

from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

N_AI_AGENT_FEATURES = 20

AI_AGENT_FEATURE_NAMES = [
    "tx_time_entropy",
    "tx_time_fourier_peak",
    "tx_hourly_std",
    "gas_price_volatility",
    "contract_diversity_hhi",
    "method_signature_repetition",
    "event_response_latency_p50",
    "swarm_correlation",
    "identical_amount_ratio",
    "nonce_gap_consistency",
    "weekend_activity_ratio",
    "first_tx_to_activity_ratio",
    "self_tx_ratio",
    "interaction_graph_depth",
    "token_approval_burst",
    "failed_tx_ratio",
    "gas_limit_variance",
    "contract_deploy_count",
    "bridge_interaction_count",
    "mev_activity_score",
]


# ----------------------------------------------------------------------------
# Primitive stats
# ----------------------------------------------------------------------------


def _shannon_entropy(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    total = sum(values)
    if total == 0:
        return 0.0
    probs = [v / total for v in values if v > 0]
    return float(-sum(p * math.log2(p) for p in probs if p > 0))


def _dominant_frequency(timestamps: np.ndarray) -> float:
    """Fourier peak strength: how 'cron-like' the tx pattern is.

    Returns dominant frequency amplitude / total energy (0..1).
    High = regular periodic pattern (bot).
    """
    if len(timestamps) < 8:
        return 0.0
    # Convert to inter-arrival intervals. The pipeline caller already hands us
    # a sorted array (compute_ai_agent_features sorts the timeline, then casts
    # — a monotonic transform), so re-sorting copies the whole array for
    # nothing; on a high-degree address that copy is gigabytes. Other callers
    # may pass unsorted input, so the contract is kept: check first, sort only
    # if needed. The comparison allocates one bool array (1 byte/elem) against
    # a sort's 8 bytes/elem plus O(n log n).
    if len(timestamps) > 1 and bool(np.all(timestamps[1:] >= timestamps[:-1])):
        sorted_ts = timestamps
    else:
        sorted_ts = np.sort(timestamps)
    diffs = np.diff(sorted_ts)
    if len(diffs) < 4 or diffs.std() == 0:
        return 0.0
    # FFT on zero-mean signal
    signal = diffs - diffs.mean()
    spectrum = np.abs(np.fft.rfft(signal))
    if len(spectrum) <= 1:
        return 0.0
    peak = spectrum[1:].max()  # skip DC component
    total = spectrum[1:].sum()
    return float(peak / total) if total > 0 else 0.0


def _cap_last_c(
    sent_tx: pd.DataFrame, recv_tx: pd.DataFrame, cap: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep the `cap` most recent rows across both frames combined.

    Endpoint cap for extremely high-degree addresses: the combined series
    length drives the Feature 0/1/2/6/10/11 timeline, and rfft cost depends
    sharply on the exact length, so the caller can size the cap to make the
    post-cap FFT length a power of two. Tie rows at the threshold timestamp
    are taken from sent_tx first, then recv_tx, in stored order —
    deterministic for identical inputs. Addresses at or under the cap never
    reach this function, so their values cannot change.
    """
    ts_sent = (
        sent_tx["ts"].astype("int64").to_numpy() if len(sent_tx) else np.empty(0, dtype=np.int64)
    )
    ts_recv = (
        recv_tx["ts"].astype("int64").to_numpy() if len(recv_tx) else np.empty(0, dtype=np.int64)
    )
    total = len(ts_sent) + len(ts_recv)
    ts_all = np.concatenate([ts_sent, ts_recv])
    kth = np.partition(ts_all, total - cap)[total - cap]
    del ts_all
    gt_sent = ts_sent > kth
    gt_recv = ts_recv > kth
    need_eq = cap - int(gt_sent.sum()) - int(gt_recv.sum())
    eq_sent = np.flatnonzero(ts_sent == kth)
    take_sent = eq_sent[: max(need_eq, 0)]
    need_eq -= len(take_sent)
    eq_recv = np.flatnonzero(ts_recv == kth)
    take_recv = eq_recv[: max(need_eq, 0)]
    keep_sent = np.sort(np.concatenate([np.flatnonzero(gt_sent), take_sent]))
    keep_recv = np.sort(np.concatenate([np.flatnonzero(gt_recv), take_recv]))
    return sent_tx.iloc[keep_sent], recv_tx.iloc[keep_recv]


# ----------------------------------------------------------------------------
# Per-address AI-agent feature computation
# ----------------------------------------------------------------------------


def compute_ai_agent_features(
    df: pd.DataFrame,
    *,
    gas_price_col: str = "gas_price",
    gas_limit_col: str = "gas",
    nonce_col: str = "nonce",
    status_col: str | None = "receipt_status",
    method_id_col: str | None = None,
    bridge_contracts: set[str] | None = None,
    process_addrs: set[str] | None = None,
    global_bucket_counts: pd.Series | None = None,
    global_bucket_counts_max: float | None = None,
    series_cap: int | None = None,
    capped_records: list[dict] | None = None,
) -> tuple[dict[str, int], np.ndarray]:
    """Compute 20 AI-agent discriminator features per address.

    Args:
        df: Ethereum transactions DataFrame with columns:
            from_addr, to_addr, value_int, ts, [gas_price, gas, nonce, receipt_status, input]
        gas_price_col: column with gas price in Wei (optional)
        gas_limit_col: column with gas limit (optional)
        nonce_col: column with tx nonce (optional)
        status_col: column with tx success/revert (optional)
        bridge_contracts: set of known bridge contract addresses (optional)
        process_addrs: if set, compute features ONLY for this subset of
            addresses (used by multiprocessing workers that hold a hash
            partition of the graph). Default None = all addresses in df.
        global_bucket_counts: pre-computed unique-senders-per-10min-bucket
            Series for Feature 7 (swarm_correlation). If supplied, skips
            local recomputation — needed when workers see only a partition.
        global_bucket_counts_max: max value of the global bucket_counts
            Series (paired with global_bucket_counts).
        series_cap: if set, an address whose combined sent+recv row count
            exceeds this value is truncated to its `series_cap` most recent
            rows before ANY feature is computed (endpoint cap).
            None (default) leaves every code path byte-identical.
        capped_records: if provided together with series_cap, one dict per
            capped address is appended (audit sidecar input).

    Returns:
        addr_to_idx: address → row index mapping
        features: (N_addresses, 20) feature matrix
    """
    bridge_contracts = bridge_contracts or set()
    if process_addrs is None:
        all_addrs = sorted(
            set(df["from_addr"].dropna().unique()) | set(df["to_addr"].dropna().unique())
        )
    else:
        all_addrs = sorted(process_addrs)
    addr_to_idx = {addr: i for i, addr in enumerate(all_addrs)}
    n = len(all_addrs)

    # Pre-group transactions by address (sender perspective — most features are about sender)
    sender_groups = df.groupby("from_addr")
    receiver_groups = df.groupby("to_addr")

    # Ensure ts is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["ts"]):
        df = df.copy()
        df["ts"] = pd.to_datetime(df["ts"], utc=True)

    # 10-minute bucket assignment — local (used by Feature 7 if caller
    # didn't pre-compute a global bucket_counts).
    df["ts_bucket"] = df["ts"].dt.floor("10min")

    if global_bucket_counts is not None and global_bucket_counts_max is not None:
        # Worker mode: caller computed bucket_counts over the FULL graph
        # in a pre-pass; using local groupby here would wrongly scope the
        # "swarm" metric to one hash partition.
        bucket_counts = global_bucket_counts
        bucket_counts_max = float(global_bucket_counts_max)
    else:
        # Pre-compute bucket → unique-senders map ONCE (was O(N²) inside the loop).
        bucket_counts = df.groupby("ts_bucket")["from_addr"].nunique()
        bucket_counts_max = float(bucket_counts.max()) if len(bucket_counts) > 0 else 0.0

    features = np.zeros((n, N_AI_AGENT_FEATURES), dtype=np.float32)

    logger.info("ai_agent_features_start", n_addresses=n)
    progress_every = max(1, n // 20)  # 20 progress checkpoints
    # Percentage checkpoints alone go silent for as long as one address takes:
    # the first checkpoint is at idx n//20, and a single very large address
    # can hold the loop at idx 0 for a long time with no output. A wall-clock
    # heartbeat names the address we are actually stuck on, which is the one
    # fact needed to diagnose a stall. Emits at most once a minute, so it
    # cannot itself be a cost.
    _hb_interval_s = 60.0
    _hb_last = time.monotonic()
    # A single address can hold hundreds of millions of rows, and then the
    # loop spends an hour inside ONE iteration — a heartbeat evaluated only
    # between iterations would stay silent exactly when it is needed. So the
    # size of an address is announced BEFORE it is processed whenever it is
    # large enough to matter; the wall-clock heartbeat covers the ordinary
    # slow-drift case. Sizes come from .groups, which the membership test
    # below materialises anyway — .indices would be a second dict of the same
    # order and is deliberately not touched.
    _hb_big_rows = 1_000_000

    for addr, idx in addr_to_idx.items():
        if idx % progress_every == 0 and idx > 0:
            logger.info(
                "ai_agent_features_progress",
                processed=idx,
                total=n,
                pct=round(100 * idx / n, 1),
            )
        _sent_n = len(sender_groups.groups.get(addr, ()))
        _recv_n = len(receiver_groups.groups.get(addr, ()))
        _now = time.monotonic()
        if _sent_n + _recv_n >= _hb_big_rows or _now - _hb_last >= _hb_interval_s:
            _hb_last = _now
            logger.info(
                "ai_agent_features_heartbeat",
                at_idx=idx,
                total=n,
                addr=addr,
                sent_rows=_sent_n,
                recv_rows=_recv_n,
            )
        sent_tx = sender_groups.get_group(addr) if addr in sender_groups.groups else pd.DataFrame()
        recv_tx = (
            receiver_groups.get_group(addr) if addr in receiver_groups.groups else pd.DataFrame()
        )

        if len(sent_tx) == 0 and len(recv_tx) == 0:
            continue

        if series_cap is not None and len(sent_tx) + len(recv_tx) > series_cap:
            _sent_before, _recv_before = len(sent_tx), len(recv_tx)
            sent_tx, recv_tx = _cap_last_c(sent_tx, recv_tx, series_cap)
            logger.info(
                "ai_agent_features_series_capped",
                addr=addr,
                sent_before=_sent_before,
                recv_before=_recv_before,
                sent_after=len(sent_tx),
                recv_after=len(recv_tx),
                series_cap=series_cap,
            )
            if capped_records is not None:
                capped_records.append(
                    {
                        "addr": addr,
                        "sent_rows_before": int(_sent_before),
                        "recv_rows_before": int(_recv_before),
                        "sent_rows_after": int(len(sent_tx)),
                        "recv_rows_after": int(len(recv_tx)),
                        "series_cap": int(series_cap),
                    }
                )

        all_tx = pd.concat([sent_tx, recv_tx]) if len(recv_tx) > 0 else sent_tx
        timestamps = all_tx["ts"].sort_values()
        ts_seconds = timestamps.astype("int64").values // 1_000_000_000

        # ---------- Feature 0: tx_time_entropy ----------
        if len(ts_seconds) > 1:
            diffs = np.diff(ts_seconds)
            bins = [0, 60, 300, 3600, 86400, 604800, np.inf]
            hist, _ = np.histogram(diffs, bins=bins)
            features[idx, 0] = _shannon_entropy(list(hist.astype(float)))

        # ---------- Feature 1: tx_time_fourier_peak ----------
        features[idx, 1] = _dominant_frequency(ts_seconds)

        # ---------- Feature 2: tx_hourly_std ----------
        hours = pd.Series(timestamps.dt.hour).value_counts(sort=False)
        if len(hours) > 0:
            full_hours = pd.Series(0, index=range(24))
            full_hours.update(hours)
            features[idx, 2] = float(full_hours.std())

        # ---------- Feature 3: gas_price_volatility ----------
        if gas_price_col in sent_tx.columns and len(sent_tx) > 2:
            gp = sent_tx[gas_price_col].dropna().astype(float)
            if len(gp) > 2 and gp.mean() > 0:
                features[idx, 3] = float(gp.std() / gp.mean())

        # ---------- Feature 4: contract_diversity_hhi ----------
        if len(sent_tx) > 0:
            counterparties = sent_tx["to_addr"].value_counts()
            if len(counterparties) > 0:
                shares = (counterparties / counterparties.sum()) ** 2
                features[idx, 4] = float(shares.sum())

        # ---------- Feature 5: method_signature_repetition ----------
        if method_id_col and method_id_col in sent_tx.columns and len(sent_tx) > 0:
            methods = sent_tx[method_id_col].value_counts()
            if len(methods) > 0:
                features[idx, 5] = float(methods.iloc[0] / methods.sum())

        # ---------- Feature 6: event_response_latency_p50 ----------
        # Proxy: median time between consecutive tx (fast = agent-like)
        if len(ts_seconds) > 1:
            diffs = np.diff(ts_seconds)
            features[idx, 6] = float(np.median(diffs))

        # ---------- Feature 7: swarm_correlation ----------
        # Proxy: how many tx of this address share timestamp buckets with other addresses
        if len(sent_tx) > 5 and bucket_counts_max > 0:
            my_buckets = sent_tx["ts_bucket"].unique()
            if len(my_buckets) > 0:
                shared = bucket_counts.reindex(my_buckets, fill_value=0)
                features[idx, 7] = float(shared.mean() / (bucket_counts_max + 1e-9))

        # ---------- Feature 8: identical_amount_ratio ----------
        if len(sent_tx) > 0:
            amounts = sent_tx["value_int"].value_counts()
            if len(amounts) > 0 and amounts.sum() > 0:
                features[idx, 8] = float(amounts.iloc[0] / amounts.sum())

        # ---------- Feature 9: nonce_gap_consistency ----------
        if nonce_col in sent_tx.columns and len(sent_tx) > 2:
            nonces = sent_tx[nonce_col].dropna().sort_values()
            if len(nonces) > 2:
                gaps = np.diff(nonces.astype(float))
                if gaps.std() == 0 and len(gaps) > 0:
                    features[idx, 9] = 1.0
                elif gaps.mean() > 0:
                    features[idx, 9] = float(1.0 / (1.0 + gaps.std() / gaps.mean()))

        # ---------- Feature 10: weekend_activity_ratio ----------
        if len(timestamps) > 0:
            weekday = timestamps.dt.dayofweek
            weekend = (weekday >= 5).sum()
            features[idx, 10] = float(weekend / len(timestamps))

        # ---------- Feature 11: first_tx_to_activity_ratio ----------
        if len(timestamps) > 1:
            age_days = (timestamps.iloc[-1] - timestamps.iloc[0]).total_seconds() / 86400
            if age_days > 0:
                features[idx, 11] = float(len(all_tx) / (age_days + 1))
            else:
                features[idx, 11] = float(len(all_tx))

        # ---------- Feature 12: self_tx_ratio ----------
        if len(sent_tx) > 0:
            self_tx = (sent_tx["to_addr"] == addr).sum()
            features[idx, 12] = float(self_tx / len(sent_tx))

        # ---------- Feature 13: interaction_graph_depth ----------
        # Approximation: unique_counterparties / log(tx_count)
        if len(sent_tx) > 0:
            uniq = sent_tx["to_addr"].nunique()
            features[idx, 13] = float(uniq / math.log(len(sent_tx) + 2))

        # ---------- Feature 14: token_approval_burst ----------
        # Proxy: ratio of first-10 tx that are to contracts (0x followed by input data) vs rest
        # Without method_id data, use heuristic: tx with value=0 likely approvals/calls
        if len(sent_tx) >= 10:
            first10 = sent_tx.iloc[:10]
            zero_val = (first10["value_int"] == 0).sum()
            features[idx, 14] = float(zero_val / 10)

        # ---------- Feature 15: failed_tx_ratio ----------
        if status_col and status_col in sent_tx.columns and len(sent_tx) > 0:
            failed = (sent_tx[status_col] == 0).sum()
            features[idx, 15] = float(failed / len(sent_tx))

        # ---------- Feature 16: gas_limit_variance ----------
        if gas_limit_col in sent_tx.columns and len(sent_tx) > 2:
            gl = sent_tx[gas_limit_col].dropna().astype(float)
            if len(gl) > 2 and gl.mean() > 0:
                features[idx, 16] = float(gl.std() / gl.mean())

        # ---------- Feature 17: contract_deploy_count ----------
        # tx to null address indicates contract deployment
        if len(sent_tx) > 0:
            null_addrs = {None, "", "0x0000000000000000000000000000000000000000"}
            deploys = sent_tx["to_addr"].isin(null_addrs).sum()
            features[idx, 17] = float(deploys)

        # ---------- Feature 18: bridge_interaction_count ----------
        if len(sent_tx) > 0 and bridge_contracts:
            bridge_tx = sent_tx["to_addr"].isin(bridge_contracts).sum()
            features[idx, 18] = float(bridge_tx)

        # ---------- Feature 19: mev_activity_score ----------
        # Proxy: gas_price anomalies (very high gas = MEV bundle attempt)
        if gas_price_col in sent_tx.columns and len(sent_tx) > 5:
            gp = sent_tx[gas_price_col].dropna().astype(float)
            if len(gp) > 5 and gp.median() > 0:
                high_gas = (gp > gp.median() * 3).sum()
                features[idx, 19] = float(high_gas / len(gp))

    logger.info(
        "ai_agent_features_computed",
        n_addresses=n,
        n_features=N_AI_AGENT_FEATURES,
    )
    return addr_to_idx, features


def normalize_features(features: np.ndarray) -> np.ndarray:
    """Min-max normalization per column (makes model size-invariant)."""
    mins = features.min(axis=0)
    maxs = features.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0
    return (features - mins) / ranges


# ----------------------------------------------------------------------------
# Combined feature builder
# ----------------------------------------------------------------------------


def combine_with_base_features(
    base_addr_to_idx: dict[str, int],
    base_features: list[list[float]] | np.ndarray,
    ai_addr_to_idx: dict[str, int],
    ai_features: np.ndarray,
) -> tuple[dict[str, int], np.ndarray]:
    """Concatenate 16 base + 20 ai_agent features into 36-dim vectors.

    Handles address mismatches by zero-filling missing addresses.
    """
    base_arr = np.asarray(base_features, dtype=np.float32)
    all_addrs = sorted(set(base_addr_to_idx.keys()) | set(ai_addr_to_idx.keys()))
    new_idx = {a: i for i, a in enumerate(all_addrs)}
    n = len(all_addrs)
    combined = np.zeros((n, base_arr.shape[1] + ai_features.shape[1]), dtype=np.float32)

    for addr, i in new_idx.items():
        if addr in base_addr_to_idx:
            combined[i, : base_arr.shape[1]] = base_arr[base_addr_to_idx[addr]]
        if addr in ai_addr_to_idx:
            combined[i, base_arr.shape[1] :] = ai_features[ai_addr_to_idx[addr]]

    return new_idx, combined
