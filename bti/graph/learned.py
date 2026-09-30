# Copyright (c) 2026 Srijan Upadhyay. All rights reserved. Proprietary — see LICENSE.
"""
Learned graph features from weekly snapshots: spectral embeddings and a temporal GraphSAGE.

Every snapshot `s` sees only edges before `s` (customer ↔ device / IP) and fraud
confirmations before `s`. A transaction on day `d` takes the latest snapshot at
or before `d`.

**Spectral embedding (no deep learning).**
- Method: a truncated SVD of the degree-normalised customer–entity
  biadjacency.
- Feature: `graph_embed_fraud_similarity`, the highest cosine similarity
  between the customer and any customer with a confirmed fraud.
- Why a similarity: embedding coordinates rotate from one snapshot to the next,
  so they are never used directly; the similarity is rotation-invariant.
- Isolated customers get 0.

**Temporal GraphSAGE** (`graph_sage_score`; needs PyTorch).
- Architecture: a two-layer bipartite mean aggregator (entities from their
  customers, then customers from their entities) over structural and
  known-fraud node features.
- Target: "this customer commits fraud in the next 30 days".
- Training data: training-window snapshots only.
- Leakage control: rows inside the training window are scored by two-fold
  temporal cross-fitting, so the downstream model never sees a score that was
  fitted on its own labels. Later windows are scored by a model trained on
  every training snapshot.
- Inductive: new customers and entities are scored from their features and
  neighbours, with no retraining.

The fitted SAGE weights are returned so a registered model can carry them, and
the nightly snapshot can apply them.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import svds

from bti.graph.temporal import confirmation_series
from bti.modeling.features import event_timestamps

LEARNED_FEATURES = ["graph_embed_fraud_similarity", "graph_sage_score"]
EMBED_DIM = 16
HORIZON_DAYS = 30
NODE_FEATURES = ["log_txns", "n_entities", "max_co_users", "log_component", "component_known_fraud",
                 "known_fraud_self"]


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _prep(df: pd.DataFrame, delay_days: int):
    ts = event_timestamps(df)
    conf = confirmation_series(df, ts, delay_days)
    cust = df["customer_id"].astype(str).to_numpy()
    edges = []
    for col, prefix in (("device_id", "d:"), ("ip_location", "i:")):
        if col in df.columns:
            v = df[col]
            ok = v.notna() & (v.astype(str).str.strip() != "")
            edges.append(pd.DataFrame({"c": cust[ok.to_numpy()], "e": prefix + v[ok].astype(str),
                                       "t": ts[ok].to_numpy()}))
    edges = pd.concat(edges, ignore_index=True).sort_values("t")
    fraud = pd.DataFrame({"c": cust, "t": ts.to_numpy(), "conf": conf.to_numpy(),
                          "y": pd.to_numeric(df.get("fraud_flag", 0), errors="coerce").fillna(0).to_numpy()})
    return ts, edges, fraud


def _snapshot(edges: pd.DataFrame, fraud: pd.DataFrame, s: pd.Timestamp):
    e = edges[edges["t"] < s].drop_duplicates(["c", "e"])
    customers = pd.Index(sorted(fraud.loc[fraud["t"] < s, "c"].unique()))
    entities = pd.Index(sorted(e["e"].unique()))
    ci, ei = customers.get_indexer(e["c"]), entities.get_indexer(e["e"])
    keep = ci >= 0
    B = sparse.csr_matrix((np.ones(keep.sum()), (ci[keep], ei[keep])), shape=(len(customers), len(entities)))
    known = set(fraud.loc[fraud["conf"] < s, "c"])
    return customers, entities, B, known


def _node_features(fraud, customers, B, known, s):
    past = fraud[fraud["t"] < s]
    txns = past.groupby("c").size().reindex(customers).fillna(0).to_numpy()
    ent_deg = np.asarray(B.sum(axis=0)).ravel()
    co_users = B.multiply(ent_deg[None, :]).max(axis=1).toarray().ravel() - 1 if B.shape[1] else np.zeros(len(customers))
    n_c = B.shape[0]
    A = sparse.bmat([[None, B], [B.T, None]]).tocsr() if B.shape[1] else sparse.csr_matrix((n_c, n_c))
    _, labels = connected_components(A, directed=False)
    comp = labels[:n_c]
    comp_size = np.bincount(comp, minlength=comp.max() + 1)[comp] if n_c else np.zeros(0)
    k = np.array([c in known for c in customers], dtype=float)
    comp_known = np.bincount(comp, weights=k, minlength=comp.max() + 1)[comp] - k if n_c else np.zeros(0)
    n_ent = np.asarray(B.sum(axis=1)).ravel()
    X = np.column_stack([np.log1p(txns), n_ent, np.clip(co_users, 0, None), np.log1p(comp_size), comp_known, k])
    return X, comp_size, k


def spectral_similarity(customers, B, known) -> np.ndarray:
    """
    Embed only the linked subgraph (entities shared by more than one customer, and their customers); everyone else
    carries no network information and gets 0. Randomised SVD is used because the full graph's spectrum is
    degenerate (thousands of isolated customer–device pairs).
    """
    from sklearn.utils.extmath import randomized_svd
    n_c = B.shape[0]
    sim = np.zeros(n_c)
    if B.shape[1] == 0 or not known:
        return sim
    de = np.asarray(B.sum(axis=0)).ravel()
    shared = np.flatnonzero(de > 1)
    if len(shared) < 2:
        return sim
    S = B[:, shared]
    rows = np.flatnonzero(np.asarray(S.sum(axis=1)).ravel() > 0)
    known_rows = np.array([i for i in rows if customers[i] in known])
    if len(known_rows) == 0 or len(rows) < 3:
        return sim
    sub = S[rows]
    dc, dsh = np.asarray(sub.sum(axis=1)).ravel(), np.asarray(sub.sum(axis=0)).ravel()
    Bn = sparse.diags(1 / np.sqrt(dc)) @ sub @ sparse.diags(1 / np.sqrt(np.maximum(dsh, 1)))
    k = max(1, min(EMBED_DIM, min(Bn.shape) - 1))
    u, sv, _ = randomized_svd(Bn.astype(float), n_components=k, random_state=0)
    emb = u * sv
    emb = emb / np.maximum(np.linalg.norm(emb, axis=1), 1e-12)[:, None]
    local = {r: j for j, r in enumerate(rows)}
    kj = np.array([local[r] for r in known_rows])
    M = emb @ emb[kj].T
    M[kj, np.arange(len(kj))] = -1                                   # exclude self
    sim[rows] = np.clip(M.max(axis=1), 0, 1)
    return sim


# ── GraphSAGE ────────────────────────────────────────────────────────────────

def _sage_model(n_in: int, hidden: int = 16):
    import torch

    class BipartiteSAGE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.ent = torch.nn.Linear(2 + n_in, hidden)
            self.cus = torch.nn.Linear(n_in + hidden, hidden)
            self.out = torch.nn.Linear(hidden, 1)

        def forward(self, Xc, Xe, Bn_ce, Bn_ec):
            he = torch.relu(self.ent(torch.cat([Xe, torch.sparse.mm(Bn_ec, Xc)], dim=1)))
            hc = torch.relu(self.cus(torch.cat([Xc, torch.sparse.mm(Bn_ce, he)], dim=1)))
            return self.out(hc).squeeze(-1)

    return BipartiteSAGE()


def _tensors(X, B, known_counts):
    import torch
    de = np.asarray(B.sum(axis=0)).ravel()
    Xe = np.column_stack([np.log1p(de), known_counts])
    dc = np.asarray(B.sum(axis=1)).ravel()

    def sp(M):
        M = M.tocoo()
        return torch.sparse_coo_tensor(np.vstack([M.row, M.col]), M.data.astype(np.float32), M.shape)
    Bn_ce = sparse.diags(1 / np.maximum(dc, 1)) @ B
    Bn_ec = sparse.diags(1 / np.maximum(de, 1)) @ B.T
    return (torch.tensor(X, dtype=torch.float32), torch.tensor(Xe, dtype=torch.float32), sp(Bn_ce), sp(Bn_ec))


def _train_sage(batches, epochs: int = 80, seed: int = 0):
    import torch
    torch.manual_seed(seed)
    model = _sage_model(len(NODE_FEATURES))
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    pos = sum(float(y.sum()) for *_, y in batches)
    neg = sum(float(len(y) - y.sum()) for *_, y in batches)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(max(neg / max(pos, 1), 1.0)))
    for _ in range(epochs):
        for tensors, y in batches:
            opt.zero_grad()
            loss = loss_fn(model(*tensors), y)
            loss.backward()
            opt.step()
    return model


def _score(model, tensors) -> np.ndarray:
    import torch
    with torch.no_grad():
        return torch.sigmoid(model(*tensors)).numpy()


def learned_features(df: pd.DataFrame, train_end: pd.Timestamp, delay_days: int = 30, freq: str = "7D",
                     use_sage: Optional[bool] = None) -> Dict:
    """
    Per-row `graph_embed_fraud_similarity` and `graph_sage_score` (NaN when torch is unavailable), plus the fitted
    SAGE state for the model artifact.
    """
    use_sage = torch_available() if use_sage is None else use_sage
    ts, edges, fraud = _prep(df, delay_days)
    snapshots = pd.date_range(ts.min().normalize() + pd.Timedelta(days=7), ts.max().normalize() + pd.Timedelta(days=1),
                              freq=freq)
    snap_data = []
    for s in snapshots:
        customers, entities, B, known = _snapshot(edges, fraud, s)
        X, comp_size, k = _node_features(fraud, customers, B, known, s)
        known_counts = np.asarray(B.T @ k).ravel() if B.shape[1] else np.zeros(0)
        future = fraud[(fraud["t"] >= s) & (fraud["t"] < s + pd.Timedelta(days=HORIZON_DAYS)) & (fraud["y"] == 1)]
        y = customers.isin(set(future["c"])).astype(np.float32)
        snap_data.append({"s": s, "customers": customers, "B": B, "known": known, "X": X,
                          "known_counts": known_counts, "y": y})

    emb = {i: spectral_similarity(d["customers"], d["B"], d["known"]) for i, d in enumerate(snap_data)}
    sage_scores, state = {}, None
    if use_sage:
        import torch
        train_idx = [i for i, d in enumerate(snap_data) if d["s"] + pd.Timedelta(days=HORIZON_DAYS) <= train_end]
        tens = {i: _tensors(d["X"], d["B"], d["known_counts"]) for i, d in enumerate(snap_data)}
        batch = lambda idx: [(tens[i], torch.tensor(snap_data[i]["y"])) for i in idx if len(snap_data[i]["y"])]
        if len(train_idx) >= 4:
            half = len(train_idx) // 2
            a, b = train_idx[:half], train_idx[half:]
            m_a, m_b = _train_sage(batch(a)), _train_sage(batch(b))
            full = _train_sage(batch(train_idx))
            for i in range(len(snap_data)):
                model = m_b if i in a else m_a if i in b else full        # cross-fitted inside training window
                sage_scores[i] = _score(model, tens[i])
            state = {"state_dict": {k: v.detach().numpy() for k, v in full.state_dict().items()},
                     "node_features": NODE_FEATURES, "horizon_days": HORIZON_DAYS,
                     "training_snapshots": len(train_idx)}

    # map each row to the latest snapshot at or before its day
    snap_times = np.array([d["s"] for d in snap_data], dtype="datetime64[ns]")
    pos = np.searchsorted(snap_times, ts.dt.floor("D").to_numpy(), side="right") - 1
    cust = df["customer_id"].astype(str).to_numpy()
    out = np.full((len(df), 2), np.nan)
    for i, d in enumerate(snap_data):
        rows = np.flatnonzero(pos == i)
        if not len(rows):
            continue
        idx = d["customers"].get_indexer(cust[rows])
        ok = idx >= 0
        out[rows[ok], 0] = emb[i][idx[ok]]
        if i in sage_scores:
            out[rows[ok], 1] = sage_scores[i][idx[ok]]
    return {"features": pd.DataFrame(out, columns=LEARNED_FEATURES, index=df.index), "sage_state": state,
            "snapshots": len(snap_data)}


def latest_scores(df: pd.DataFrame, sage_state: Optional[Dict], delay_days: int = 30,
                  at: Optional[pd.Timestamp] = None) -> Dict[str, tuple]:
    """Spectral similarity and SAGE score per customer at one snapshot (the nightly live snapshot)."""
    ts, edges, fraud = _prep(df, delay_days)
    at = at or ts.max().normalize() + pd.Timedelta(days=1)
    customers, entities, B, known = _snapshot(edges, fraud, at)
    X, _, k = _node_features(fraud, customers, B, known, at)
    sim = spectral_similarity(customers, B, known)
    sage = np.full(len(customers), np.nan)
    if sage_state and torch_available() and len(customers):
        import torch
        model = _sage_model(len(NODE_FEATURES))
        model.load_state_dict({name: torch.tensor(v) for name, v in sage_state["state_dict"].items()})
        known_counts = np.asarray(B.T @ k).ravel() if B.shape[1] else np.zeros(0)
        sage = _score(model, _tensors(X, B, known_counts))
    return {c: (float(sim[i]), float(sage[i])) for i, c in enumerate(customers)}
