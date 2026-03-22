"""
AEDE — Adaptive Experimental Decision Engine
=============================================
A decision-theoretic framework for self-driving labs that integrates
principled stopping criteria, cost-aware experiment selection, and
causal post-campaign attribution.

Quick start:
    from aede import run_campaign
    from aede.modules.problem_framing import ProblemSpec, Objective, BudgetSpec

    spec = ProblemSpec(
        name="my_campaign",
        objectives=[
            Objective(name="binding_affinity", direction="maximize"),
            Objective(name="toxicity_score", direction="minimize"),
        ],
        budget=BudgetSpec(experiments=100),
        benchmark="vlmop2",
    )
    results = run_campaign(spec, verbose=True)
"""

from .campaign import run_campaign
from .modules.problem_framing import (
    ProblemSpec, Objective, HardConstraint, SoftConstraint,
    BudgetSpec, StoppingCriterion, load_spec_from_yaml, load_spec_from_dict,
)
from .modules.trace_logger import DecisionTraceLogger
from .modules.insight_engine import InsightEngine

__version__ = "0.1.0"
__all__ = [
    "run_campaign",
    "ProblemSpec",
    "Objective",
    "HardConstraint",
    "SoftConstraint",
    "BudgetSpec",
    "StoppingCriterion",
    "load_spec_from_yaml",
    "load_spec_from_dict",
    "DecisionTraceLogger",
    "InsightEngine",
]
