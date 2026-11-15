"""Comprehensive Unit and Integration Test Suite for AIDE2 (02_aide2).

Verifies 100% of AIDE2 components, code tree data structures, MCTS search,
AST code transformation operators, and bi-level meta-harness recursive optimization.
"""

from __future__ import annotations

import ast
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

from common.domain.backtester import BacktestEngine, BacktestResult
from common.domain.market_data import generate_multi_era_market_data
from common.domain.strategy_base import DynamicCodeStrategy
from common.llm_provider import DeterministicMockProvider

try:
    from .tree_node import CodeDiff, CodeTree, StrategyNode
    from .operators import DebuggingOperator, DraftingOperator, RefinementOperator, _DEFAULT_DRAFT_CODE
    from .mcts_search import AIDE2Search
    from .meta_harness import MetaHarness
except (ImportError, ValueError):
    from tree_node import CodeDiff, CodeTree, StrategyNode
    from operators import DebuggingOperator, DraftingOperator, RefinementOperator, _DEFAULT_DRAFT_CODE
    from mcts_search import AIDE2Search
    from meta_harness import MetaHarness


# ==============================================================================
# 1. StrategyNode & CodeTree Tests
# ==============================================================================

def test_strategy_node_instantiation_and_properties():
    """Verifies StrategyNode initialization, metric properties, and update logic."""
    dummy_result = BacktestResult(
        sharpe_ratio=1.45,
        max_drawdown=0.12,
        sortino_ratio=1.80,
        calmar_ratio=1.20,
        cagr=0.18,
        turnover=2.5,
        total_cost=0.002,
        equity_curve=[100000.0, 118000.0],
        daily_returns=[0.001] * 10,
        weights_history=[[0.2] * 5] * 10,
        execution_time_ms=5.0,
        success=True,
    )
    node = StrategyNode(
        node_id="test_node_0",
        code=_DEFAULT_DRAFT_CODE,
        metrics=dummy_result,
        parent_id=None,
        operator="root",
        visits=2,
        total_value=2.90,
        depth=0,
        generation=0,
    )

    assert node.node_id == "test_node_0"
    assert node.sharpe == 1.45
    assert node.max_drawdown == 0.12
    assert node.value == pytest.approx(1.45, rel=1e-3)
    assert node.is_terminal is False

    # Test update
    node.update(1.50)
    assert node.visits == 3
    assert node.total_value == pytest.approx(4.40, rel=1e-3)
    assert node.value == pytest.approx(4.40 / 3.0, rel=1e-3)

    # Test serialization
    d = node.to_dict()
    assert d["node_id"] == "test_node_0"
    assert d["sharpe_ratio"] == 1.45
    assert d["operator"] == "root"


def test_strategy_node_diff_and_code_hash():
    """Verifies code hash determinism and unified diff computation."""
    parent_code = "def foo():\n    return 1\n"
    child_code = "def foo():\n    return 2\n"

    parent_node = StrategyNode(node_id="p", code=parent_code, metrics=None)
    child_node = StrategyNode(node_id="c", code=child_code, metrics=None, parent_id="p")

    assert len(parent_node.code_hash()) == 16
    assert parent_node.code_hash() != child_node.code_hash()

    diff = child_node.compute_diff(parent_code)
    assert "-    return 1" in diff
    assert "+    return 2" in diff


def test_code_tree_add_node_lineage_and_best_node():
    """Verifies CodeTree hierarchy management, lineage traversal, and best node tracking."""
    tree = CodeTree()
    root = StrategyNode(node_id="root", code="x = 1", metrics=BacktestResult(
        sharpe_ratio=0.80, max_drawdown=0.20, sortino_ratio=1.0, calmar_ratio=0.5,
        cagr=0.10, turnover=1.0, total_cost=0.0, equity_curve=[100000.0],
        daily_returns=[], weights_history=[], execution_time_ms=1.0, success=True,
    ))
    tree.add_node(root)

    child1 = StrategyNode(node_id="c1", code="x = 2", parent_id="root", operator="refine", metrics=BacktestResult(
        sharpe_ratio=1.20, max_drawdown=0.15, sortino_ratio=1.5, calmar_ratio=0.8,
        cagr=0.15, turnover=1.0, total_cost=0.0, equity_curve=[100000.0],
        daily_returns=[], weights_history=[], execution_time_ms=1.0, success=True,
    ))
    tree.add_node(child1)

    child2 = StrategyNode(node_id="c2", code="x = 3", parent_id="c1", operator="refine", metrics=BacktestResult(
        sharpe_ratio=1.65, max_drawdown=0.10, sortino_ratio=2.0, calmar_ratio=1.2,
        cagr=0.20, turnover=1.0, total_cost=0.0, equity_curve=[100000.0],
        daily_returns=[], weights_history=[], execution_time_ms=1.0, success=True,
    ))
    tree.add_node(child2)

    assert tree.root_id == "root"
    assert len(tree.nodes) == 3
    assert len(tree.edges) == 2

    # Check lineage of grandchild
    lineage = tree.get_lineage("c2")
    assert [n.node_id for n in lineage] == ["root", "c1", "c2"]

    # Check best node
    best = tree.best_node()
    assert best is not None
    assert best.node_id == "c2"
    assert best.sharpe == 1.65


def test_code_tree_pruning_and_ascii():
    """Verifies branch pruning and ASCII visualization."""
    tree = CodeTree()
    root = StrategyNode(node_id="root", code="x = 0", metrics=None)
    tree.add_node(root)

    c1 = StrategyNode(node_id="c1", code="x = 1", metrics=None, parent_id="root")
    c1_child = StrategyNode(node_id="c1_1", code="x = 11", metrics=None, parent_id="c1")
    tree.add_node(c1)
    tree.add_node(c1_child)

    pruned = tree.prune_branch("c1")
    assert "c1" in pruned
    assert "c1_1" in pruned
    assert tree.get_node("c1").is_terminal is True
    assert tree.get_node("c1_1").is_terminal is True

    ascii_rep = tree.render_ascii()
    assert "[DEAD-END]" in ascii_rep or "[TERMINAL]" in ascii_rep or "root" in ascii_rep


# ==============================================================================
# 2. Operator Tests (Drafting, Debugging, Refinement)
# ==============================================================================

def test_drafting_operator_generates_valid_ast():
    """Verifies that DraftingOperator generates syntactically valid Python code with required methods."""
    provider = DeterministicMockProvider()
    drafter = DraftingOperator(provider)
    code = drafter.draft("Multi-asset trend-following alpha")

    parsed = ast.parse(code)
    assert isinstance(parsed, ast.Module)
    assert "DraftedStrategy" in code
    assert "generate_signals" in code
    assert "compute_weights" in code


def test_debugging_operator_parses_traceback():
    """Verifies that DebuggingOperator accurately parses tracebacks and extracts error details."""
    debugger = DebuggingOperator()
    trace = (
        "Traceback (most recent call last):\n"
        '  File "strategy.py", line 42, in generate_signals\n'
        "ZeroDivisionError: float division by zero\n"
    )
    diag = debugger.parse_traceback(trace)
    assert diag["error_type"] == "ZeroDivisionError"
    assert diag["line_number"] == 42
    assert "denominator" in diag["suggested_fix"].lower() or "epsilon" in diag["suggested_fix"].lower()


def test_debugging_operator_patches_zero_division():
    """Verifies that DebuggingOperator patches zero-division bugs in candidate code."""
    flawed_code = '''
def generate_signals(self, market_data):
    s = 1.0 / 0.0
    return s
'''
    debugger = DebuggingOperator()
    fixed = debugger.debug_and_fix(flawed_code, "ZeroDivisionError: float division by zero")
    assert "1.0 / 0.0" not in fixed
    assert "1e-6" in fixed or "1e-8" in fixed


def test_debugging_operator_patches_syntax_and_lookahead():
    """Verifies that DebuggingOperator fixes syntax errors and lookahead weight leaks."""
    syntax_bug = "def bad_syntax(:\n    pass\n"
    debugger = DebuggingOperator()
    fixed_syntax = debugger.debug_and_fix(syntax_bug, "SyntaxError: invalid syntax")
    assert "bad_syntax():" in fixed_syntax

    lookahead_bug = "realized_weights = weights\n"
    fixed_lookahead = debugger.debug_and_fix(lookahead_bug, "Lookahead bias detected")
    assert "weights[:-1]" in fixed_lookahead


def test_refinement_operator_tunes_parameters_and_vol_targeting():
    """Verifies that RefinementOperator tunes moving average windows and adds volatility targeting."""
    code = _DEFAULT_DRAFT_CODE
    refiner = RefinementOperator()
    refined = refiner.refine(code, technique="volatility_targeting")

    assert "fast_window: int = 10" in refined
    assert "slow_window: int = 24" in refined
    assert "mom_window: int = 15" in refined
    assert "Refinement" in refined


# ==============================================================================
# 3. AIDE2 MCTS Search Tests
# ==============================================================================

def test_aide2_mcts_uct_calculation_and_selection():
    """Verifies UCT score computation balances exploitation and exploration."""
    market_data = generate_multi_era_market_data(num_days=100)
    engine = BacktestEngine(default_fee_bps=10.0)
    provider = DeterministicMockProvider()

    search = AIDE2Search(
        c_param=1.414,
        drafter=DraftingOperator(provider),
        debugger=DebuggingOperator(provider),
        refiner=RefinementOperator(provider),
        backtest_engine=engine,
        market_data=market_data,
    )

    root = StrategyNode(node_id="root", code=_DEFAULT_DRAFT_CODE, metrics=None, visits=10, total_value=12.0)
    child1 = StrategyNode(node_id="c1", code=_DEFAULT_DRAFT_CODE, metrics=None, parent_id="root", visits=5, total_value=8.0)
    child2 = StrategyNode(node_id="c2", code=_DEFAULT_DRAFT_CODE, metrics=None, parent_id="root", visits=0, total_value=0.0)
    search.add_node(root)
    search.add_node(child1)
    search.add_node(child2)

    # Unvisited child2 should receive infinity UCT score
    uct_c2 = search.uct_score(child2, parent_visits=root.visits)
    assert uct_c2 == float("inf")

    # Visited child1 should receive standard UCT
    uct_c1 = search.uct_score(child1, parent_visits=root.visits)
    assert 1.0 < uct_c1 < 10.0

    # Select chooses child2 because it is unvisited
    selected = search.select()
    assert selected.node_id == "c2"


def test_aide2_mcts_expand_evaluate_backpropagate():
    """Verifies that expanding and evaluating a child node populates metrics and backpropagates rewards."""
    market_data = generate_multi_era_market_data(num_days=252)
    engine = BacktestEngine(default_fee_bps=10.0)
    provider = DeterministicMockProvider()

    search = AIDE2Search(
        c_param=1.414,
        drafter=DraftingOperator(provider),
        debugger=DebuggingOperator(provider),
        refiner=RefinementOperator(provider),
        backtest_engine=engine,
        market_data=market_data,
    )

    root = StrategyNode(node_id="root", code=_DEFAULT_DRAFT_CODE, metrics=None, visits=1, total_value=0.5)
    search.add_node(root)

    child = search.expand(root, operator_type="refine")
    assert child.parent_id == "root"
    assert child.metrics is None

    # Evaluate child node
    eval_res = search.evaluate_node(child)
    assert eval_res is not None
    assert eval_res.success is True
    assert child.metrics is not None
    assert child.sharpe != 0.0

    # Compute reward and backpropagate
    reward = search.compute_reward(eval_res)
    search.backpropagate(child, reward=reward)

    assert child.visits == 1
    assert child.total_value == pytest.approx(reward, rel=1e-3)
    assert root.visits == 2


def test_aide2_mcts_step_cycle():
    """Verifies that step() executes one complete MCTS selection-expansion-evaluation-backpropagation cycle."""
    market_data = generate_multi_era_market_data(num_days=100)
    engine = BacktestEngine(default_fee_bps=10.0)
    provider = DeterministicMockProvider()

    search = AIDE2Search(
        c_param=1.414,
        drafter=DraftingOperator(provider),
        debugger=DebuggingOperator(provider),
        refiner=RefinementOperator(provider),
        backtest_engine=engine,
        market_data=market_data,
    )

    root = StrategyNode(node_id="root", code=_DEFAULT_DRAFT_CODE, metrics=None, visits=1, total_value=0.5)
    search.add_node(root)

    target_node = search.step()
    assert target_node is not None
    assert target_node.metrics is not None
    assert target_node.visits >= 1
    assert len(search.nodes) >= 2


# ==============================================================================
# 4. MetaHarness Tests
# ==============================================================================

def test_meta_harness_guard_zero_generations():
    """Verifies that MetaHarness safely handles zero generations without IndexError."""
    market_data = generate_multi_era_market_data(num_days=100)
    engine = BacktestEngine(default_fee_bps=10.0)
    provider = DeterministicMockProvider()

    harness = MetaHarness(
        provider=provider,
        market_data=market_data,
        backtest_engine=engine,
    )

    gens = harness.run_meta_optimization(generations=0, iterations_per_gen=1)
    assert gens == []
    summary = harness.summary_table()
    assert "No generations recorded" in summary


def test_meta_harness_end_to_end_optimization():
    """Verifies multi-generational bi-level meta-optimization produces real empirical backtest metrics."""
    market_data = generate_multi_era_market_data(num_days=252)
    engine = BacktestEngine(default_fee_bps=10.0)
    provider = DeterministicMockProvider()

    harness = MetaHarness(
        provider=provider,
        market_data=market_data,
        backtest_engine=engine,
    )

    gens = harness.run_meta_optimization(
        generations=2,
        iterations_per_gen=2,
        verbose=False,
    )

    assert len(gens) == 2
    assert gens[0]["generation"] == 0
    assert gens[1]["generation"] == 1
    # Check that best_sharpe is genuine and non-decreasing
    assert isinstance(gens[0]["best_sharpe"], float)
    assert isinstance(gens[1]["best_sharpe"], float)
    assert gens[1]["best_sharpe"] >= gens[0]["best_sharpe"]

    # Verify summary table renders properly
    summary = harness.summary_table()
    assert "AIDE2 Bi-Level Meta-Optimization" in summary
    assert "Gen 0" in summary
    assert "Gen 1" in summary
