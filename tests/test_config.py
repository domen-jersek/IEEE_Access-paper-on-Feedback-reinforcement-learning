"""Tests for config module invariants."""
import pytest
from src.config import ProjectPaths, LiftConfig, RoutingConfig, GatingConfig, EvalConfig


def test_project_paths_exist():
    paths = ProjectPaths()
    assert paths.root.exists()
    assert paths.data_processed.exists()
    assert paths.configs.exists()
    assert paths.results.exists()


def test_lift_config_to_dict():
    cfg = LiftConfig.laplace()
    d = cfg.to_dict()
    assert d["name"] == "laplace"
    assert "alpha" in d
    assert "beta" in d
    assert "multiplier" in d
    assert "cap" in d


def test_routing_config_to_dict():
    cfg = RoutingConfig.global_()
    d = cfg.to_dict()
    assert d["name"] == "global"
    assert d["w_class"] == 0.0
    assert d["w_team"] == 0.0

    cfg2 = RoutingConfig.intersection()
    assert cfg2.name == "categorical_intersection"


def test_eval_config_hash_is_stable():
    cfg1 = EvalConfig(
        experiment_id="test",
        lift=LiftConfig.laplace(),
        routing=RoutingConfig.global_(),
        gating=GatingConfig.none(),
        generator_model="test/model",
        judge_model="test/judge",
        seed=42,
    )
    cfg2 = EvalConfig(
        experiment_id="test",
        lift=LiftConfig.laplace(),
        routing=RoutingConfig.global_(),
        gating=GatingConfig.none(),
        generator_model="test/model",
        judge_model="test/judge",
        seed=42,
    )
    assert cfg1.config_hash == cfg2.config_hash


def test_eval_config_hash_differs():
    cfg1 = EvalConfig(
        experiment_id="test",
        lift=LiftConfig.laplace(),
        routing=RoutingConfig.global_(),
        gating=GatingConfig.none(),
        generator_model="test/model",
        judge_model="test/judge",
        seed=42,
    )
    cfg2 = EvalConfig(
        experiment_id="test",
        lift=LiftConfig.tanh(),
        routing=RoutingConfig.global_(),
        gating=GatingConfig.none(),
        generator_model="test/model",
        judge_model="test/judge",
        seed=42,
    )
    assert cfg1.config_hash != cfg2.config_hash


def test_default_methods():
    from src.config import DEFAULT_METHODS
    for name, cfg in DEFAULT_METHODS.items():
        assert isinstance(cfg, EvalConfig)
        assert cfg.experiment_id
        assert cfg.generator_model
        assert cfg.judge_model