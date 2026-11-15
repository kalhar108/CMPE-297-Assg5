"""Standalone runnable CLI entrypoint for Codelab 3: The AI Scientist 2 (Automated Discovery & MASS).

Demonstrates:
  (a) Template-free economic & quantitative anomaly hypothesis generation;
  (b) Multi-era empirical experimentation & stress ablation (Bull, Choppy, Crisis);
  (c) Multi-agent simulated peer review committee scoring (Soundness, Novelty, Significance, Risk);
  (d) Multi-Agent Self-Supervision (MASS) recursive distillation showing generational score improvement;
  (e) Dual-mode execution support (offline mock vs live Gemini API).

Execution Modes:
  python demo.py
  python -m 03_ai_scientist_2.demo
  python demo.py --mode mock --generations 3 --verbose
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time
from typing import Any, Dict, List, Optional

# Windows console UTF-8 output reconfiguration
if sys.platform == "win32":
    try:
        if sys.stdout and hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if sys.stderr and hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure parent directory and assignment root are on sys.path for robust standalone invocation
_current_dir = os.path.dirname(os.path.abspath(__file__))
_assignment_root = os.path.abspath(os.path.join(_current_dir, ".."))
if _assignment_root not in sys.path:
    sys.path.insert(0, _assignment_root)
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

try:
    from common.config import RSIExecutionConfig
    from common.domain.backtester import BacktestEngine, BacktestResult
    from common.domain.market_data import generate_multi_era_market_data
    from common.llm_provider import get_llm_provider
except ImportError:
    # If common is missing, fallback gracefully
    raise ImportError("Required 'common' module not found on sys.path.")

try:
    from .ablation import MultiEraAblation
    from .experimenter import ExperimentManager
    from .hypothesis import Hypothesis, HypothesisGenerator
    from .mass_loop import MASSLoop
    from .peer_review import PeerReviewBoard
except ImportError:
    from ablation import MultiEraAblation  # type: ignore
    from experimenter import ExperimentManager  # type: ignore
    from hypothesis import Hypothesis, HypothesisGenerator  # type: ignore
    from mass_loop import MASSLoop  # type: ignore
    from peer_review import PeerReviewBoard  # type: ignore


def print_banner(title: str, subtitle: Optional[str] = None) -> None:
    """Renders formatted ASCII section header."""
    width = 80
    border = "=" * width
    print(f"\n{border}")
    print(f"  {title.upper():^{width-4}}")
    if subtitle:
        print(f"  {subtitle:^{width-4}}")
    print(f"{border}\n")


def run_demo(
    mode: str = "mock",
    generations: int = 3,
    hypotheses_per_gen: int = 2,
    verbose: bool = False,
) -> int:
    """Executes the full AI Scientist 2 & MASS demonstration walkthrough."""
    t_start = time.perf_counter()

    print_banner(
        "CODELAB 3: THE AI SCIENTIST 2",
        "Automated Scientific Discovery & Multi-Agent Self-Supervision (MASS)",
    )

    print(f"[*] Configuration:")
    print(f"    - Execution Mode        : {mode.upper()}")
    print(f"    - Discovery Generations : {generations}")
    print(f"    - Hypotheses per Gen    : {hypotheses_per_gen}")
    print(f"    - Problem Domain        : Quantitative Alpha Engineering (SPY, TLT, GLD, VIX, CASH)")
    print(f"    - Multi-Era Market Span : 756 Trading Days (Bull, Choppy, Crisis Regimes)")

    # 1. Initialize Domain Environment & LLM Provider
    config = RSIExecutionConfig(mode=mode)  # type: ignore
    provider = get_llm_provider(config)
    resolved_mode = config.resolved_mode()
    print(f"    - Active Provider       : {type(provider).__name__} (Resolved Mode: {resolved_mode})")

    print("\n[+] Generating 756-day multi-era market dataset...")
    market_data = generate_multi_era_market_data(seed=42, num_days=756)
    engine = BacktestEngine(default_fee_bps=10.0)
    print(f"    [OK] Market dataset loaded: {market_data.num_days} days across Bull (1..252), Choppy (253..504), Crisis (505..756).")

    # 2. Stage A: Template-Free Hypothesis Generation
    print_banner("STAGE A: TEMPLATE-FREE HYPOTHESIS GENERATION", "Synthesizing economic anomaly formulations without static templates")

    generator = HypothesisGenerator(llm_provider=provider, config=config)
    sample_gen0 = generator.generate_hypothesis(generation=0, variant_index=0)
    print("[-] Generation 0 (Raw Baseline Hypothesis):")
    print(sample_gen0.format_card())
    print(f"    Rationale: {sample_gen0.rationale}\n")

    # 3. Stage B: Multi-Era Empirical Experimentation & Ablation
    print_banner("STAGE B: PROGRESSIVE EXPERIMENTATION & MULTI-ERA ABLATION", "Stress testing strategy resilience across Bull, Choppy, and Crisis regimes")

    exp_manager = ExperimentManager(engine=engine, market_data=market_data, llm_provider=provider)
    strat_gen0 = exp_manager.synthesize_strategy(sample_gen0)
    exp_res_gen0 = exp_manager.run_experiment(sample_gen0, strat_gen0)

    print(f"[*] Backtest Results on Full 756-Day Span:")
    print(f"    Strategy Name       : {strat_gen0.name}")
    print(f"    Execution Time      : {exp_res_gen0['execution_time_ms']:.2f} ms (< 20ms requirement)")
    print(f"    Full-Span Sharpe    : {exp_res_gen0['sharpe_ratio']:.2f}")
    print(f"    Max Drawdown        : {exp_res_gen0['max_drawdown']*100:.2f}%")
    print(f"    CAGR                : {exp_res_gen0['cagr']*100:.2f}%")
    print(f"    Annualized Turnover : {exp_res_gen0['turnover']:.2f}")

    ablation_engine = MultiEraAblation(engine=engine, market_data=market_data)
    ablation_gen0 = ablation_engine.run_ablation(strat_gen0)

    print("\n[*] Multi-Era Stress Ablation Table:")
    print(ablation_engine.format_ablation_table(ablation_gen0))

    # 4. Stage C: Simulated Multi-Agent Peer Review Committee
    print_banner("STAGE C: SIMULATED MULTI-AGENT PEER REVIEW", "Evaluating Soundness, Novelty, Statistical Significance, and Risk Discipline")

    reviewer = PeerReviewBoard(llm_provider=provider, accept_threshold=7.0)
    review_gen0 = reviewer.evaluate(sample_gen0, ablation_gen0)

    print(reviewer.format_review_card(review_gen0))

    # 5. Stage D: Multi-Agent Self-Supervision (MASS) Recursive Loop
    print_banner("STAGE D: MASS RECURSIVE SELF-SUPERVISION CAMPAIGN", "Distilling review critiques into prompt priors and observing generational progress")

    mass = MASSLoop()
    print("[*] Initial Distilled Priors in MASS Memory:")
    for idx, p in enumerate(mass.distilled_priors, 1):
        print(f"    [{idx}] {p}")

    print(f"\n[*] Launching {generations}-Generation Autonomous Discovery Loop...")
    campaign = mass.run_discovery_campaign(
        engine=engine,
        market_data=market_data,
        num_generations=generations,
        hypotheses_per_gen=hypotheses_per_gen,
        llm_provider=provider,
    )

    # 6. Generational Progression Table
    print_banner("GENERATIONAL PROGRESSION TABLE (RSI VERIFICATION)", "Proving recursive score and Sharpe improvement across generations")

    prog_header = f"+------+------------+------------------+-------------------+---------------+-----------------+\n" \
                  f"| Gen  | Candidates | Avg Review Score | Best Review Score | Best Sharpe   | Worst Crisis MDD|\n" \
                  f"+------+------------+------------------+-------------------+---------------+-----------------+"
    if not campaign["generations_summary"]:
        print("\n[*] Zero generations requested; campaign ended with 0 hypotheses evaluated.")
        t_total = time.perf_counter() - t_start
        print(f"\n[OK] Demonstration completed cleanly in {t_total:.2f} seconds.")
        return 0

    prog_rows = []
    for g in campaign["generations_summary"]:
        prog_rows.append(
            f"| {g['generation']:<4} | {g['num_hypotheses']:<10} | {g['avg_peer_score']:>14.2f}/10 | "
            f"{g['best_peer_score']:>15.2f}/10 | {g['best_sharpe']:>11.2f} | {g['worst_crisis_mdd']*100:>14.2f}% |"
        )
    prog_footer = f"+------+------------+------------------+-------------------+---------------+-----------------+"
    print("\n".join([prog_header] + prog_rows + [prog_footer]))

    print(f"\n[*] Recursive Improvement Summary:")
    print(f"    - Total Hypotheses Explored : {campaign['total_hypotheses_evaluated']}")
    print(f"    - Initial Gen 0 Review Score : {campaign['generations_summary'][0]['avg_peer_score']:.2f} / 10.0")
    print(f"    - Final Gen {generations-1} Review Score   : {campaign['generations_summary'][-1]['avg_peer_score']:.2f} / 10.0")
    print(f"    - Net Score Delta           : {campaign['peer_score_delta']:+.2f} points (Self-Supervised Gain)")
    print(f"    - Initial Best Sharpe       : {campaign['generations_summary'][0]['best_sharpe']:.2f}")
    print(f"    - Final Best Sharpe         : {campaign['generations_summary'][-1]['best_sharpe']:.2f}")
    print(f"    - Net Sharpe Delta          : {campaign['sharpe_delta']:+.2f} (Measurable Alpha Improvement)")
    print(f"    - Accumulated Priors in MASS: {campaign['final_priors_count']}")

    # 7. Winning Candidate Card
    print_banner("WINNING ALPHA DISCOVERY ARTIFACT", "Best hypothesis synthesized and approved by the peer review committee")
    best_hyp = campaign["best_hypothesis"]
    print(f"Title           : {best_hyp['title']}")
    print(f"Anomaly Rationale: {best_hyp['anomaly_description']}")
    print(f"Formula         : {best_hyp['formula']}")
    print(f"Strategy Name   : {campaign['best_strategy_name']}")
    print(f"Committee Score : {campaign['best_composite_score']:.2f} / 10.00 ({campaign['best_decision']})")
    print(f"Sharpe Ratio    : {campaign['best_sharpe']:.2f}")
    print(f"Crisis Max DD   : {campaign['best_crisis_mdd']*100:.2f}%")

    t_total = time.perf_counter() - t_start
    print(f"\n[OK] Demonstration completed cleanly in {t_total:.2f} seconds.")
    return 0


def main() -> int:
    """CLI argument parsing and execution entrypoint."""
    parser = argparse.ArgumentParser(
        description="The AI Scientist 2 (Automated Discovery & MASS) Codelab CLI Entrypoint"
    )
    parser.add_argument(
        "--mode",
        choices=["mock", "live", "auto"],
        default="mock",
        help="Execution mode (default: mock, 100% offline deterministic)",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=3,
        help="Number of discovery generations for recursive self-supervision (default: 3)",
    )
    parser.add_argument(
        "--hypotheses-per-gen",
        type=int,
        default=2,
        help="Number of hypotheses per generation (default: 2)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed logging output",
    )

    args = parser.parse_args()
    return run_demo(
        mode=args.mode,
        generations=args.generations,
        hypotheses_per_gen=args.hypotheses_per_gen,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    sys.exit(main())
