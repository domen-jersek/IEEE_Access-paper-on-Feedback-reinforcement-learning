"""
Central configuration for the IEEE Access paper evaluation.
Frozen dataclasses, ProjectPaths, and experiment parameter definitions.
No hardcoded paths or magic numbers anywhere else in the codebase.
"""
from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union


@dataclass(frozen=True)
class ProjectPaths:
    root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])
    data_raw: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1] / "data" / "raw")
    data_processed: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1] / "data" / "processed")
    results: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1] / "results")

    @property
    def configs(self) -> Path:
        return Path(__file__).resolve().parents[1] / "configs"

    @property
    def source_csv(self) -> Path:
        return self.root.parent / "notebooks" / "tickets_large_first_reply_label.csv"

    @property
    def source_db(self) -> Path:
        return self.root.parent / "notebooks" / "tickets.db"

    @property
    def dataset_parquet(self) -> Path:
        return self.data_processed / "dataset.parquet"

    @property
    def manifest_json(self) -> Path:
        return self.data_processed / "dataset_manifest.json"

    @property
    def taxonomy_csv(self) -> Path:
        return self.data_processed / "taxonomy.csv"

    @property
    def groups_json(self) -> Path:
        return self.data_processed / "procedure_groups.json"

    @property
    def splits_dir(self) -> Path:
        return self.data_processed / "splits"


# --- Frozen experiment configs ---

@dataclass(frozen=True)
class LiftConfig:
    name: str
    alpha: float = 1.0
    beta: float = 1.0
    multiplier: float = 0.80
    cap: float = 0.20
    positive_only: bool = False
    sensitivity: float = 3.5
    lcb_k: float = 1.0
    # --- P1.5 calibration extensions (defaults reproduce the SIKDD behaviour) ---
    # prior_strength: total pseudo-count kappa of the Beta prior used by `laplace_eb`
    # (alpha = kappa * p_bar, beta = kappa * (1 - p_bar), p_bar = empirical scope mean).
    prior_strength: float = 2.0
    # scale_mode: "absolute" adds the lift directly to the retrieval score (SIKDD);
    # "pool_std" expresses the lift in units of the candidate pool's score std
    # (cap  <->  pool_lambda * std), which makes lifts comparable across retrievers.
    scale_mode: str = "absolute"
    pool_lambda: float = 1.0

    _NEW_DEFAULTS = {"prior_strength": 2.0, "scale_mode": "absolute", "pool_lambda": 1.0}

    def to_dict(self) -> dict:
        d = {
            "name": self.name,
            "alpha": self.alpha,
            "beta": self.beta,
            "multiplier": self.multiplier,
            "cap": self.cap,
            "positive_only": self.positive_only,
            "sensitivity": self.sensitivity,
            "lcb_k": self.lcb_k,
        }
        # New keys are emitted only when non-default so that config hashes of the
        # already-completed SIKDD replication runs are unchanged.
        for k, default in self._NEW_DEFAULTS.items():
            v = getattr(self, k)
            if v != default:
                d[k] = v
        return d

    @staticmethod
    def laplace() -> LiftConfig:
        return LiftConfig(
            name="laplace", alpha=1.0, beta=1.0, multiplier=0.80, cap=0.20
        )

    @staticmethod
    def laplace_eb(prior_strength: float = 2.0, scale_mode: str = "absolute",
                   pool_lambda: float = 1.0) -> LiftConfig:
        """Empirical-Bayes centred Laplace lift (P1.5)."""
        return LiftConfig(
            name="laplace_eb", multiplier=0.80, cap=0.20,
            prior_strength=prior_strength, scale_mode=scale_mode, pool_lambda=pool_lambda,
        )

    @staticmethod
    def tanh() -> LiftConfig:
        return LiftConfig(
            name="tanh", multiplier=0.80, cap=0.20, sensitivity=3.5
        )

    @staticmethod
    def bayesian_lcb() -> LiftConfig:
        return LiftConfig(
            name="bayesian_lcb", multiplier=0.80, cap=0.20, lcb_k=1.0
        )


@dataclass(frozen=True)
class RoutingConfig:
    name: str
    w_global: float = 1.0
    w_class: float = 0.0
    w_team: float = 0.0
    require_semantic: bool = False
    # --- P3 granularity extensions (defaults keep legacy routings unchanged) ---
    # "blend": enhanced = score + sum_s w_s * lift_s  (per-scope lifts, incl. intersection)
    w_intersection: float = 0.0
    # "backoff": first scope in `backoff_order` whose evidence n >= min_evidence
    backoff_order: tuple = ("intersection", "team", "class", "global")
    min_evidence: float = 3.0

    _NEW_DEFAULTS = {
        "w_intersection": 0.0,
        "backoff_order": ("intersection", "team", "class", "global"),
        "min_evidence": 3.0,
    }

    @staticmethod
    def global_() -> RoutingConfig:
        return RoutingConfig(name="global", w_global=1.0, w_class=0.0, w_team=0.0)

    @staticmethod
    def team_only() -> RoutingConfig:
        return RoutingConfig(name="categorical", w_global=0.0, w_class=0.0, w_team=1.0)

    @staticmethod
    def class_only() -> RoutingConfig:
        return RoutingConfig(name="categorical", w_global=0.0, w_class=1.0, w_team=0.0)

    @staticmethod
    def intersection() -> RoutingConfig:
        return RoutingConfig(name="categorical_intersection", w_global=0.0, w_class=1.0, w_team=1.0)

    @staticmethod
    def backoff(order: tuple = ("intersection", "team", "class", "global"),
                min_evidence: float = 3.0) -> RoutingConfig:
        return RoutingConfig(name="backoff", w_global=0.0, backoff_order=tuple(order),
                             min_evidence=min_evidence)

    @staticmethod
    def blend(w_global: float = 0.0, w_class: float = 0.0, w_team: float = 0.0,
              w_intersection: float = 0.0) -> RoutingConfig:
        return RoutingConfig(name="blend", w_global=w_global, w_class=w_class,
                             w_team=w_team, w_intersection=w_intersection)

    def to_dict(self) -> dict:
        d = {
            "name": self.name,
            "w_global": self.w_global,
            "w_class": self.w_class,
            "w_team": self.w_team,
            "require_semantic": self.require_semantic,
        }
        for k, default in self._NEW_DEFAULTS.items():
            v = getattr(self, k)
            if v != default:
                d[k] = list(v) if isinstance(v, tuple) else v
        return d


@dataclass(frozen=True)
class GatingConfig:
    name: str
    faiss_ceiling: float = 1.0
    fuzzy_low: float = 0.0
    fuzzy_high: float = 0.0
    miracle_tau: float = 0.0

    @staticmethod
    def none() -> GatingConfig:
        return GatingConfig(name="none")

    @staticmethod
    def static(ceiling: float = 0.656) -> GatingConfig:
        return GatingConfig(name="static", faiss_ceiling=ceiling)

    @staticmethod
    def learned() -> GatingConfig:
        return GatingConfig(name="learned")

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "faiss_ceiling": self.faiss_ceiling,
            "fuzzy_low": self.fuzzy_low,
            "fuzzy_high": self.fuzzy_high,
            "miracle_tau": self.miracle_tau,
        }


@dataclass(frozen=True)
class EvalConfig:
    experiment_id: str
    lift: LiftConfig
    routing: RoutingConfig
    gating: GatingConfig
    generator_model: str
    judge_model: str
    embedding_model: str = "all-MiniLM-L6-v2"
    top_k: int = 5
    search_k: int = 100
    temperature: float = 0.0
    regime: str = "random"
    seed: int = 42
    # --- P2: retriever used for the candidate pool (see src/retrieval/retrievers.py) ---
    retriever: str = "dense_minilm"
    # --- P1.2: extra answer metrics ("core" = cosine+rouge_l+length as in SIKDD runs)
    metric_set: str = "core"

    _NEW_DEFAULTS = {"retriever": "dense_minilm", "metric_set": "core"}

    def to_dict(self) -> dict:
        d = {
            "experiment_id": self.experiment_id,
            "lift": self.lift.to_dict(),
            "routing": self.routing.to_dict(),
            "gating": self.gating.to_dict(),
            "generator_model": self.generator_model,
            "judge_model": self.judge_model,
            "embedding_model": self.embedding_model,
            "top_k": self.top_k,
            "search_k": self.search_k,
            "temperature": self.temperature,
            "regime": self.regime,
            "seed": self.seed,
        }
        for k, default in self._NEW_DEFAULTS.items():
            v = getattr(self, k)
            if v != default:
                d[k] = v
        return d

    @property
    def config_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]


# Scope keys used in feedback_scores[candidate_id][...]
SCOPE_GLOBAL = "global"
SCOPE_ORDER = ("intersection", "team", "class", "global")

# Retriever identifiers accepted by --retriever (P2)
RETRIEVERS = (
    "dense_minilm",      # all-MiniLM-L6-v2 FAISS (SIKDD default)
    "bm25",              # lexical BM25Okapi on title + description
    "hybrid_rrf",        # RRF(dense_minilm, bm25)
    "dense_bge",         # BAAI/bge-small-en-v1.5 FAISS (independent embedder family)
    "hybrid_bge_rrf",    # RRF(dense_bge, bm25)
    "ce_rerank",         # cross-encoder/ms-marco-MiniLM-L-6-v2 re-ranking the dense_minilm pool
    "ce_hybrid_rerank",  # cross-encoder re-ranking the hybrid_rrf pool
)
ALT_EMBEDDERS = {
    "minilm": "all-MiniLM-L6-v2",
    "bge": "BAAI/bge-small-en-v1.5",
}
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# Independent-family embedder used for the alternative answer metric / proxy (P1.2)
ALT_METRIC_EMBEDDER = "BAAI/bge-base-en-v1.5"


# --- Default method definitions ---

GENERATOR_MODEL = "openai/gpt-5.6-luna"
JUDGE_MODEL_ID = "openai/gpt-5.6-luna"

DEFAULT_METHODS = {
    "baseline": EvalConfig(
        experiment_id="baseline",
        lift=LiftConfig(name="none", multiplier=0.0, cap=0.0),
        routing=RoutingConfig(name="none"),
        gating=GatingConfig.none(),
        generator_model=GENERATOR_MODEL,
        judge_model=JUDGE_MODEL_ID,
    ),
    "M1_global": EvalConfig(
        experiment_id="M1_global",
        lift=LiftConfig.laplace(),
        routing=RoutingConfig.global_(),
        gating=GatingConfig.none(),
        generator_model=GENERATOR_MODEL,
        judge_model=JUDGE_MODEL_ID,
    ),
    "M2_team": EvalConfig(
        experiment_id="M2_team_only",
        lift=LiftConfig.laplace(),
        routing=RoutingConfig.team_only(),
        gating=GatingConfig.none(),
        generator_model=GENERATOR_MODEL,
        judge_model=JUDGE_MODEL_ID,
    ),
    "M3_class": EvalConfig(
        experiment_id="M3_class_only",
        lift=LiftConfig.laplace(),
        routing=RoutingConfig.class_only(),
        gating=GatingConfig.none(),
        generator_model=GENERATOR_MODEL,
        judge_model=JUDGE_MODEL_ID,
    ),
    "M4_intersection": EvalConfig(
        experiment_id="M4_intersection",
        lift=LiftConfig.laplace(),
        routing=RoutingConfig.intersection(),
        gating=GatingConfig.none(),
        generator_model=GENERATOR_MODEL,
        judge_model=JUDGE_MODEL_ID,
    ),
    # --- P3 granularity methods (new; not part of the SIKDD replication) ---
    "M5_backoff": EvalConfig(
        experiment_id="M5_backoff",
        lift=LiftConfig.laplace(),
        routing=RoutingConfig.backoff(),
        gating=GatingConfig.none(),
        generator_model=GENERATOR_MODEL,
        judge_model=JUDGE_MODEL_ID,
    ),
    # Blend weights are placeholders; 12_learn_blend.py writes the learned weights to
    # results/blend/learned_weights.json and 04_evaluate.py --blend-weights loads them.
    "M6_blend": EvalConfig(
        experiment_id="M6_blend",
        lift=LiftConfig.laplace(),
        routing=RoutingConfig.blend(w_global=0.0, w_class=0.25, w_team=0.5, w_intersection=0.5),
        gating=GatingConfig.none(),
        generator_model=GENERATOR_MODEL,
        judge_model=JUDGE_MODEL_ID,
    ),
}