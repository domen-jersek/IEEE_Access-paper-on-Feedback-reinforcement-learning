"""
Vectorised retrieval-only evaluation over cached candidate pools (P2 ladder, P3 blend).

Workflow
--------
1. `collect_pools(retriever, ...)` runs the retriever once per query (search_k
   candidates, LOO self-exclusion) and caches the pool to parquet.
2. `PoolTensor` turns pools + feedback evidence + offline usefulness into dense
   arrays so that any (lift, routing, scale) configuration can be evaluated for
   all queries with a few numpy operations — no API calls, sub-second per config.

The vectorised lift/routing math mirrors `src/feedback/lift.py` and
`src/retrieval/search.py`; `tests/test_ladder_consistency.py` checks agreement.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..config import LiftConfig, RoutingConfig
from ..feedback.loader import FeedbackBundle
from ..retrieval.search import scope_key, SCOPES
from .proxy import ReplySimilarity

log = logging.getLogger(__name__)
SCOPE_IDX = {s: i for i, s in enumerate(SCOPES)}  # global, class, team, intersection


# ---------------------------------------------------------------------------
# Pool collection
# ---------------------------------------------------------------------------

def collect_pools(retriever, queries: pd.DataFrame, encoder, search_k: int, cache_path: Optional[Path] = None,
                  id_to_index=None) -> pd.DataFrame:
    """
    Long frame: query_id, seq_id, faiss_score, faiss_rank for the top-`search_k`
    of each query (self excluded). Cached to parquet when `cache_path` is given.
    `queries` must contain seq_id, Title_anon, Description_anon.
    """
    if cache_path and cache_path.exists():
        df = pd.read_parquet(cache_path)
        if set(df["query_id"].unique()) >= set(queries["seq_id"]):
            return df[df["query_id"].isin(set(queries["seq_id"]))]
    rows = []
    for _, q in queries.iterrows():
        qid = str(q["seq_id"])
        title = str(q["Title_anon"])
        desc = str(q.get("Description_anon", "") or "")
        text = f"{title}\n{desc}" if desc else title
        emb = encoder.encode_ticket(title, desc)
        self_idx = id_to_index(qid) if id_to_index else None
        pool = retriever.search_pool(text, emb, search_k, {self_idx} if self_idx is not None else set())
        for r in pool.itertuples(index=False):
            rows.append((qid, str(r.seq_id), float(r.faiss_score), int(r.faiss_rank)))
    df = pd.DataFrame(rows, columns=["query_id", "seq_id", "faiss_score", "faiss_rank"])
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
    return df


# ---------------------------------------------------------------------------
# Dense tensor
# ---------------------------------------------------------------------------

@dataclass
class PoolTensor:
    query_ids: list[str]
    cand_ids: np.ndarray          # (Q, K) object
    score: np.ndarray             # (Q, K) float, -inf where padded
    mask: np.ndarray              # (Q, K) bool
    pos: np.ndarray               # (Q, K, 4)
    neg: np.ndarray               # (Q, K, 4)
    prior: np.ndarray             # (Q, 4) prior mean per scope
    useful: dict                  # sim_tag -> (Q, K) reply similarity to reference
    same_reply: np.ndarray        # (Q, K)
    same_group: np.ndarray        # (Q, K)
    query_class: list[str]
    query_team: list[str]

    @property
    def n(self) -> np.ndarray:
        return self.pos + self.neg

    # ---- lifts -----------------------------------------------------------
    def lifts(self, cfg: LiftConfig) -> np.ndarray:
        """(Q, K, 4) bounded lift per scope."""
        pos, neg, n = self.pos, self.neg, self.n
        with np.errstate(divide="ignore", invalid="ignore"):
            if cfg.name == "laplace":
                p = (pos + cfg.alpha) / (n + cfg.alpha + cfg.beta)
                lift = (p - 0.5) * np.minimum(1.0, n / 2.0) * cfg.multiplier
            elif cfg.name == "laplace_eb":
                kappa = max(cfg.prior_strength, 1e-9)
                pbar = np.clip(self.prior, 1e-6, 1 - 1e-6)[:, None, :]
                a = kappa * pbar
                p_post = (pos + a) / (n + kappa)
                lift = (p_post - pbar) * cfg.multiplier
            elif cfg.name == "tanh":
                lift = np.tanh(((pos - neg) * 0.5) / cfg.sensitivity) * cfg.cap
            elif cfg.name == "bayesian_lcb":
                p = (pos + 1.0) / (n + 2.0)
                sigma = np.sqrt(p * (1 - p) / (n + 1.0))
                lift = ((p - 0.5) - cfg.lcb_k * sigma) * np.minimum(1.0, n / 2.0) * cfg.multiplier
            elif cfg.name == "none":
                return np.zeros_like(pos)
            else:
                raise ValueError(cfg.name)
        lift = np.where(n > 0, lift, 0.0)
        if cfg.positive_only:
            lift = np.maximum(lift, 0.0)
        return np.clip(lift, -cfg.cap, cfg.cap)

    def legacy_lift(self, cfg: LiftConfig, routing: RoutingConfig) -> np.ndarray:
        """(Q, K) lift for SIKDD routings (single aggregated pos/neg -> lift)."""
        g, c, t, i = (SCOPE_IDX[s] for s in ("global", "class", "team", "intersection"))
        if routing.name == "global":
            pos, neg = self.pos[..., g] * routing.w_global, self.neg[..., g] * routing.w_global
            prior = self.prior[:, g]
        elif routing.name == "categorical":
            pos = routing.w_global * self.pos[..., g] + routing.w_class * self.pos[..., c] + routing.w_team * self.pos[..., t]
            neg = routing.w_global * self.neg[..., g] + routing.w_class * self.neg[..., c] + routing.w_team * self.neg[..., t]
            if routing.w_team > 0 and routing.w_class == 0 and routing.w_global == 0:
                prior = self.prior[:, t]
            elif routing.w_class > 0 and routing.w_team == 0 and routing.w_global == 0:
                prior = self.prior[:, c]
            else:
                prior = self.prior[:, g]
        elif routing.name == "categorical_intersection":
            pos, neg, prior = self.pos[..., i], self.neg[..., i], self.prior[:, i]
        else:
            raise ValueError(routing.name)
        tmp = PoolTensor(self.query_ids, self.cand_ids, self.score, self.mask, pos[..., None], neg[..., None],
                         prior[:, None], self.useful, self.same_reply, self.same_group, self.query_class, self.query_team)
        return tmp.lifts(cfg)[..., 0]

    def routed_lift(self, cfg: LiftConfig, routing: RoutingConfig) -> np.ndarray:
        """(Q, K) lift under any routing."""
        if routing.name in ("global", "categorical", "categorical_intersection"):
            return self.legacy_lift(cfg, routing)
        L = self.lifts(cfg)
        if routing.name == "backoff":
            n = self.n
            out = np.zeros(L.shape[:2])
            done = np.zeros(L.shape[:2], dtype=bool)
            for s in routing.backoff_order:
                j = SCOPE_IDX[s]
                take = (~done) & (n[..., j] >= routing.min_evidence)
                out = np.where(take, L[..., j], out)
                done |= take
            return out
        if routing.name == "blend":
            w = np.array([routing.w_global, routing.w_class, routing.w_team, routing.w_intersection])
            return L @ w
        if routing.name == "none":
            return np.zeros(L.shape[:2])
        raise ValueError(routing.name)

    def blend_lift(self, L: np.ndarray, w: np.ndarray) -> np.ndarray:
        return L @ np.asarray(w, float)

    # ---- re-ranking and metrics ------------------------------------------
    def scale_factor(self, cfg: LiftConfig) -> np.ndarray:
        """(Q,) multiplicative factor turning bounded lifts into score units."""
        if cfg.scale_mode == "pool_std" and cfg.cap > 0:
            sc = np.where(self.mask, self.score, np.nan)
            std = np.nanstd(sc, axis=1, ddof=1)
            return (cfg.pool_lambda * np.nan_to_num(std)) / cfg.cap
        return np.ones(len(self.query_ids))

    def rerank(self, lift: np.ndarray, factor: Optional[np.ndarray] = None, k: int = 5) -> np.ndarray:
        """Indices (Q, k) into K of the top-k by enhanced score (ties -> lower base rank)."""
        if factor is None:
            factor = np.ones(len(self.query_ids))
        enhanced = self.score + lift * factor[:, None]
        enhanced = np.where(self.mask, enhanced, -np.inf)
        return np.argsort(-enhanced, axis=1, kind="stable")[:, :k]

    def metrics(self, top: np.ndarray, sim_tag: str) -> dict[str, np.ndarray]:
        """Per-query proxy metrics for a (Q, k) index array."""
        rows = np.arange(len(self.query_ids))[:, None]
        u = self.useful[sim_tag][rows, top]
        sr = self.same_reply[rows, top]
        sg = self.same_group[rows, top]
        return {
            "proxy_top1": u[:, 0], "proxy_best5": u.max(axis=1), "proxy_mean5": u.mean(axis=1),
            "same_reply_hit1": sr[:, 0], "same_reply_hit5": sr.max(axis=1),
            "group_hit1": sg[:, 0], "group_hit5": sg.max(axis=1),
        }

    def baseline_top(self, k: int = 5) -> np.ndarray:
        return self.rerank(np.zeros_like(self.score), None, k)

    def coverage(self) -> np.ndarray:
        """(Q, 4) fraction of pool candidates with evidence per scope."""
        has = (self.n > 0) & self.mask[..., None]
        return has.sum(axis=1) / np.maximum(self.mask.sum(axis=1), 1)[:, None]


def build_pool_tensor(pools: pd.DataFrame, queries: pd.DataFrame, bundle: FeedbackBundle, sims: dict[str, ReplySimilarity],
                      per_query_loo: bool = False, k_pool: Optional[int] = None) -> PoolTensor:
    """
    pools    : long frame from collect_pools
    queries  : rows with seq_id, intent_class, Team->Name (order defines Q)
    bundle   : FeedbackBundle (already excluding the evaluated split, or use per_query_loo=True for TRAIN)
    """
    qmeta = queries.set_index("seq_id")
    qids = [q for q in queries["seq_id"].astype(str) if q in set(pools["query_id"])]
    grouped = {q: g.sort_values("faiss_rank") for q, g in pools.groupby("query_id")}
    K = k_pool or max(len(g) for g in grouped.values())
    Q = len(qids)
    cand = np.empty((Q, K), dtype=object)
    score = np.full((Q, K), -np.inf)
    mask = np.zeros((Q, K), dtype=bool)
    pos = np.zeros((Q, K, 4))
    neg = np.zeros((Q, K, 4))
    prior = np.full((Q, 4), 0.5)
    useful = {tag: np.zeros((Q, K)) for tag in sims}
    same_reply = np.zeros((Q, K))
    same_group = np.zeros((Q, K))
    qcls, qteam = [], []
    any_sim = next(iter(sims.values())) if sims else None
    for qi, qid in enumerate(qids):
        g = grouped[qid].head(K)
        cls = str(qmeta.loc[qid, "intent_class"] or "other")
        team = str(qmeta.loc[qid, "Team->Name"] or "Unknown")
        qcls.append(cls)
        qteam.append(team)
        scores_map = bundle.minus_query(qid) if per_query_loo else bundle.scores
        for j, s in enumerate(SCOPES):
            prior[qi, j] = bundle.prior_mean(scope_key(s, cls, team))
        for ci, r in enumerate(g.itertuples(index=False)):
            cid = str(r.seq_id)
            cand[qi, ci] = cid
            score[qi, ci] = r.faiss_score
            mask[qi, ci] = True
            cdata = scores_map.get(cid, {})
            for j, s in enumerate(SCOPES):
                e = cdata.get(scope_key(s, cls, team))
                if e:
                    pos[qi, ci, j] = e.get("pos", 0.0)
                    neg[qi, ci, j] = e.get("neg", 0.0)
            for tag, sim in sims.items():
                useful[tag][qi, ci] = sim.s(qid, cid)
            if any_sim is not None:
                same_reply[qi, ci] = float(any_sim.reply_hash[cid] == any_sim.reply_hash[qid])
                same_group[qi, ci] = float(any_sim.group_hash[cid] == any_sim.group_hash[qid])
    return PoolTensor(qids, cand, score, mask, pos, neg, prior, useful, same_reply, same_group, qcls, qteam)
