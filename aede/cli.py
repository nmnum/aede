"""
AEDE Command-Line Interface
============================
Usage examples:

  # Run with a YAML config
  python -m aede.cli --config configs/vlmop2_quick.yaml

  # Run with inline overrides
  python -m aede.cli --config configs/molecular_admet.yaml --seed 123 --experiments 150

  # Save trace to a file (enables post-run dashboard)
  python -m aede.cli --config configs/vlmop2_quick.yaml --db results/run1.db

  # Then launch dashboard pointing at that DB
  streamlit run app.py
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(
        description="AEDE — Adaptive Experimental Decision Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--db", default=":memory:", help="SQLite DB path for trace log (default: in-memory)")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--experiments", type=int, default=None, help="Override budget experiments")
    parser.add_argument("--mc", type=int, default=64, help="MC samples for EPI estimation (default: 64)")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-iteration output")
    parser.add_argument("--production", action="store_true",
                        help="Use BoTorch surrogate if available (requires torch/botorch)")

    args = parser.parse_args()

    from aede.modules.problem_framing import load_spec_from_yaml
    from aede import run_campaign
    import dataclasses

    spec = load_spec_from_yaml(args.config)

    # Apply CLI overrides by rebuilding spec with modified fields
    overrides = {}
    if args.seed is not None:
        overrides["random_seed"] = args.seed
    if args.experiments is not None:
        budget = dataclasses.replace(spec.budget, experiments=args.experiments)
        overrides["budget"] = budget

    if overrides:
        spec = dataclasses.replace(spec, **overrides)

    if args.production:
        try:
            import torch
            import botorch
            print(f"[CLI] BoTorch backend active (torch {torch.__version__})")
        except ImportError:
            print("[CLI] WARNING: --production flag set but torch/botorch not installed. "
                  "Falling back to sklearn GP.\n"
                  "Install: pip install torch botorch gpytorch")

    os.makedirs(os.path.dirname(args.db), exist_ok=True) if args.db != ":memory:" and os.path.dirname(args.db) else None

    results = run_campaign(
        spec,
        db_path=args.db,
        verbose=not args.quiet,
        n_mc=args.mc,
    )

    print(f"\nTrace saved to: {args.db}" if args.db != ":memory:" else "\n(trace in-memory only — use --db to persist)")
    print(f"Campaign hash:  {results['campaign_hash']}")

    if args.db != ":memory:":
        print(f"\nLaunch dashboard:\n  streamlit run app.py")


if __name__ == "__main__":
    main()
