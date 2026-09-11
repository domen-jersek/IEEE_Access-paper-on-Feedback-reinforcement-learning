"""Tests for lift formula correctness."""
import pytest
from src.config import LiftConfig
from src.feedback.lift import compute_lift, laplace, tanh, bayesian_lcb


def test_laplace_no_observations():
    cfg = LiftConfig.laplace()
    assert compute_lift(0, 0, cfg) == 0.0


def test_laplace_all_positive():
    cfg = LiftConfig.laplace()
    lift = compute_lift(10, 0, cfg)
    assert lift > 0
    assert lift <= cfg.cap


def test_laplace_all_negative():
    cfg = LiftConfig.laplace()
    lift = compute_lift(0, 10, cfg)
    assert lift < 0
    assert lift >= -cfg.cap


def test_laplace_balanced():
    cfg = LiftConfig.laplace()
    lift = compute_lift(5, 5, cfg)
    assert abs(lift) < 0.01


def test_laplace_single_observation():
    cfg = LiftConfig.laplace()
    lift = compute_lift(1, 0, cfg)
    assert 0 < lift < cfg.cap


def test_laplace_cap():
    cfg = LiftConfig.laplace()
    cfg = LiftConfig(name="laplace", alpha=1.0, beta=1.0, multiplier=5.0, cap=0.20)
    lift = compute_lift(100, 0, cfg)
    assert lift == cfg.cap


def test_laplace_positive_only():
    cfg = LiftConfig(name="laplace", positive_only=True, alpha=1.0, beta=1.0, multiplier=0.80, cap=0.20)
    lift = compute_lift(0, 100, cfg)
    assert lift == 0.0


def test_tanh():
    cfg = LiftConfig.tanh()
    lift = compute_lift(10, 0, cfg)
    assert 0 < lift <= cfg.cap

    cfg_big = LiftConfig(name="tanh", sensitivity=0.1, cap=0.5)
    lift_big = compute_lift(100, 0, cfg_big)
    assert abs(lift_big - cfg_big.cap) < 0.01


def test_bayesian_lcb():
    cfg = LiftConfig.bayesian_lcb()
    lift_no_obs = compute_lift(0, 0, cfg)
    assert lift_no_obs == 0.0

    lift_pos = compute_lift(10, 0, cfg)
    assert lift_pos > 0

    lift_neg = compute_lift(0, 10, cfg)
    assert lift_neg < 0


def test_none_lift():
    cfg = LiftConfig(name="none")
    assert compute_lift(10, 0, cfg) == 0.0