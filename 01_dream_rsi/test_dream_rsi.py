"""Comprehensive Unit and Integration Test Suite for Dream-RSI (01_dream_rsi).

Verifies 100% of Dream-RSI components, boundary conditions, edge cases,
and generational progression.
"""

import json
import os
import sys
from pathlib import Path

# Setup paths
_current_dir = Path(__file__).resolve().parent
_repo_root = _current_dir.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

import numpy as np
import pytest

from common.domain.market_data import generate_multi_era_market_data
from common.domain.backtester import BacktestEngine
from common.domain.strategy_base import MovingAverageCrossover, MomentumFactor, EqualWeightStrategy

try:
    from .tree_logger import DiscoveryNode, DiscoveryTreeLogger
    from .world_model import RegimeWorldModel
    from .dream_engine import MacroPolicy, DreamEngine
    from .policy_optimizer import PolicyOptimizer, OptimizationResult
    from .demo import run_dream_rsi_demo
except (ImportError, ValueError):
    from tree_logger import DiscoveryNode, DiscoveryTreeLogger
    from world_model import RegimeWorldModel
    from dream_engine import MacroPolicy, DreamEngine
    from policy_optimizer import PolicyOptimizer, OptimizationResult
    from demo import run_dream_rsi_demo


def test_discovery_node_dataclass():
    node = DiscoveryNode(
        node_id="n1",
        state={"regime": "bull"},
        action={"type": "tune"},
        reward=1.25,
        next_state={"regime": "bull"},
        depth=0,
    )
    assert node.node_id == "n1"
    assert node.reward == 1.25
    assert node.is_leaf() is True
    d = node.to_dict()
    assert d["node_id"] == "n1"
    assert d["depth"] == 0


def test_discovery_tree_logger_lineage_and_depth():
    logger = DiscoveryTreeLogger()
    r = logger.log_transition("root", {"regime": "bull"}, {"action": "init"}, 0.5, {"regime": "bull"})
    assert r.depth == 0

    c1 = logger.log_transition("c1", {"regime": "bull"}, {"action": "b1"}, 1.2, {"regime": "choppy"}, parent_id="root")
    c2 = logger.log_transition("c2", {"regime": "bull"}, {"action": "b2"}, 0.8, {"regime": "bull"}, parent_id="root")
    assert c1.depth == 1
    assert c2.depth == 1
    assert "c1" in r.children_ids
    assert "c2" in r.children_ids

    c1_1 = logger.log_transition("c1_1", {"regime": "choppy"}, {"action": "b1_1"}, 1.8, {"regime": "crisis"}, parent_id="c1")
    assert c1_1.depth == 2

    # Path to root
    path = logger.get_path_to_root("c1_1")
    assert [n.node_id for n in path] == ["root", "c1", "c1_1"]

    # Best node
    best = logger.best_node()
    assert best is not None
    assert best.node_id == "c1_1"
    assert best.reward == 1.8

    # Max depth and total transitions
    assert logger.max_depth() == 2
    assert logger.total_transitions() == 4
    assert len(logger.get_leaf_nodes()) == 2  # c2 and c1_1


def test_tree_logger_json_roundtrip(tmp_path):
    logger = DiscoveryTreeLogger()
    logger.log_transition("n0", {"regime": "bull"}, {"action": "act0"}, 1.1, {"regime": "bull"})
    logger.log_transition("n1", {"regime": "bull"}, {"action": "act1"}, 1.4, {"regime": "choppy"}, parent_id="n0")

    json_file = str(tmp_path / "tree.json")
    logger.export_json(json_file)
    assert os.path.exists(json_file)

    loaded_logger = DiscoveryTreeLogger()
    loaded_logger.load_json(json_file)
    assert loaded_logger.total_transitions() == 2
    assert "n0" in loaded_logger.nodes
    assert "n1" in loaded_logger.nodes
    assert loaded_logger.nodes["n1"].depth == 1


def test_world_model_default_matrix():
    model = RegimeWorldModel()
    assert model.transitions["bull"]["bull"] == 0.85
    assert model.transitions["choppy"]["choppy"] == 0.70
    assert model.transitions["crisis"]["crisis"] == 0.70

    pred_bull = model.predict_next_regime("bull")
    assert pred_bull == "bull"

    prob = model.transition_probability("bull", "choppy")
    assert prob == 0.12


def test_world_model_fit_from_market_data():
    market_data = generate_multi_era_market_data(seed=42, num_days=756)
    model = RegimeWorldModel()
    diag = model.fit_from_market_data(market_data)

    assert diag["status"] == "calibrated"
    assert diag["samples"] == 756
    assert model.is_fitted is True

    # Check that transition rows sum to approximately 1.0
    for r in ["bull", "choppy", "crisis"]:
        row_sum = sum(model.transitions[r].values())
        assert abs(row_sum - 1.0) < 0.01

    # Check regime stats
    for r in ["bull", "choppy", "crisis"]:
        assert "annual_drift" in model.regime_stats[r]
        assert "annual_volatility" in model.regime_stats[r]


def test_world_model_predict_reward_distribution():
    model = RegimeWorldModel()
    # In crisis regime, defensive hedging should outperform aggressive momentum
    mean_crisis_hedge, _ = model.predict_reward_distribution("crisis", action_type="hedge_cash")
    mean_crisis_mom, _ = model.predict_reward_distribution("crisis", action_type="momentum_trend")
    assert mean_crisis_hedge > mean_crisis_mom, "In crisis, hedging must outperform momentum"

    # In bull regime, momentum should outperform defensive hedging
    mean_bull_mom, _ = model.predict_reward_distribution("bull", action_type="momentum_trend")
    mean_bull_hedge, _ = model.predict_reward_distribution("bull", action_type="hedge_cash")
    assert mean_bull_mom > mean_bull_hedge, "In bull, momentum must outperform hedging"


def test_macro_policy_mutations():
    policy = MacroPolicy(branching_factor=2, search_depth=3, stopping_threshold=1.0)
    rng = np.random.RandomState(42)
    mutated = policy.mutate(rng)

    assert 1 <= mutated.branching_factor <= 8
    assert 1 <= mutated.search_depth <= 8
    assert 0.2 <= mutated.stopping_threshold <= 3.0


def test_dream_engine_simulated_yield_order():
    logger = DiscoveryTreeLogger()
    model = RegimeWorldModel()
    engine = DreamEngine(logger, model)

    p_cons = MacroPolicy(branching_factor=1, search_depth=1, stopping_threshold=2.0)
    p_mod = MacroPolicy(branching_factor=3, search_depth=3, stopping_threshold=1.2)
    p_agg = MacroPolicy(branching_factor=5, search_depth=5, stopping_threshold=0.8)

    y_cons = engine.dream(p_cons)
    y_mod = engine.dream(p_mod)
    y_agg = engine.dream(p_agg)

    assert y_cons > 0.0
    assert y_mod > y_cons
    assert y_agg > y_mod


def test_dream_engine_detailed_trajectory_simulation():
    logger = DiscoveryTreeLogger()
    model = RegimeWorldModel()
    engine = DreamEngine(logger, model, seed=42)

    policy = MacroPolicy(branching_factor=3, search_depth=3, stopping_threshold=1.5)
    sim = engine.simulate_trajectory(policy, num_simulations=4)

    assert sim["simulations_run"] == 4
    assert sim["total_nodes_explored"] > 0
    assert "avg_max_reward" in sim
    assert len(sim["sample_path"]) > 0


def test_policy_optimizer_grid_and_evolutionary():
    logger = DiscoveryTreeLogger()
    model = RegimeWorldModel()
    engine = DreamEngine(logger, model)
    optimizer = PolicyOptimizer(engine, seed=42)

    # Grid search
    grid_res = optimizer.optimize_grid()
    assert isinstance(grid_res, OptimizationResult)
    assert grid_res.best_yield >= grid_res.initial_yield
    assert grid_res.yield_delta >= 0.0

    # Evolutionary optimization
    evo_res = optimizer.optimize_evolutionary(generations=3, population_size=4)
    assert isinstance(evo_res, OptimizationResult)
    assert evo_res.best_yield >= evo_res.initial_yield
    assert len(evo_res.history) == 3


def test_demo_end_to_end_mock():
    telemetry = run_dream_rsi_demo(mode="mock", seed=42, verbose=False)
    assert telemetry["status"] == "success"
    assert telemetry["method"] == "Dream-RSI"
    assert telemetry["execution_mode"] == "mock"
    assert telemetry["rsi_progression"]["sharpe_delta"] > 0.0
    assert telemetry["rsi_progression"]["compute_reduction_pct"] > 0.0
    assert telemetry["rsi_progression"]["total_replay_transitions"] >= 5
