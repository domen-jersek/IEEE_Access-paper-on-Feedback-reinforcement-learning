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

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "alpha": self.alpha,
            "beta": self.beta,
            "multiplier": self.multiplier,
            "cap": self.cap,
            "positive_only": self.positive_only,
            "sensitivity": self.sensitivity,
            "lcb_k": self.lcb_k,
        }

    @staticmethod
    def laplace() -> LiftConfig:
        return LiftConfig(
            name="laplace", alpha=1.0, beta=1.0, multiplier=0.80, cap=0.20
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

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "w_global": self.w_global,
            "w_class": self.w_class,
            "w_team": self.w_team,
            "require_semantic": self.require_semantic,
        }


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

    def to_dict(self) -> dict:
        return {
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

    @property
    def config_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]


# --- Default method definitions ---

DEFAULT_METHODS = {
    "baseline": EvalConfig(
        experiment_id="baseline",
        lift=LiftConfig(name="none", multiplier=0.0, cap=0.0),
        routing=RoutingConfig(name="none"),
        gating=GatingConfig.none(),
        generator_model="openai/gpt-4o-mini-2024-07-18",
        judge_model="openai/gpt-4o-mini-2024-07-18",
    ),
    "M1_global": EvalConfig(
        experiment_id="M1_global",
        lift=LiftConfig.laplace(),
        routing=RoutingConfig.global_(),
        gating=GatingConfig.none(),
        generator_model="openai/gpt-4o-mini-2024-07-18",
        judge_model="openai/gpt-4o-mini-2024-07-18",
    ),
    "M2_team": EvalConfig(
        experiment_id="M2_team_only",
        lift=LiftConfig.laplace(),
        routing=RoutingConfig.team_only(),
        gating=GatingConfig.none(),
        generator_model="openai/gpt-4o-mini-2024-07-18",
        judge_model="openai/gpt-4o-mini-2024-07-18",
    ),
    "M3_class": EvalConfig(
        experiment_id="M3_class_only",
        lift=LiftConfig.laplace(),
        routing=RoutingConfig.class_only(),
        gating=GatingConfig.none(),
        generator_model="openai/gpt-4o-mini-2024-07-18",
        judge_model="openai/gpt-4o-mini-2024-07-18",
    ),
    "M4_intersection": EvalConfig(
        experiment_id="M4_intersection",
        lift=LiftConfig.laplace(),
        routing=RoutingConfig.intersection(),
        gating=GatingConfig.none(),
        generator_model="openai/gpt-4o-mini-2024-07-18",
        judge_model="openai/gpt-4o-mini-2024-07-18",
    ),
}