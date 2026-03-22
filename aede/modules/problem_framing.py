"""
Module 0: Problem Framing Layer
================================
Defines, validates, and locks the campaign configuration.
Uses stdlib dataclasses — no pydantic dependency.
"""

from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional
import numpy as np


@dataclass(frozen=True)
class Objective:
    name: str
    direction: str = "maximize"   # "maximize" or "minimize"
    unit: str = "dimensionless"
    weight: float = 1.0

    def __post_init__(self):
        assert self.direction in ("maximize", "minimize"), f"direction must be maximize/minimize, got {self.direction}"
        assert self.weight > 0, "weight must be positive"


@dataclass(frozen=True)
class HardConstraint:
    property: str
    min: Optional[float] = None
    max: Optional[float] = None
    equals: Optional[bool] = None
    description: str = ""


@dataclass(frozen=True)
class SoftConstraint:
    property: str
    max_violation_probability: float = 0.2
    cost_multiplier: float = 1.5
    description: str = ""

    def __post_init__(self):
        assert 0.0 < self.max_violation_probability < 1.0, "max_violation_probability must be in (0, 1)"


@dataclass(frozen=True)
class BudgetSpec:
    experiments: int = 200
    per_iteration_cap: int = 10
    cost_ceiling: float = float("inf")

    def __post_init__(self):
        assert self.experiments >= 1
        assert self.per_iteration_cap >= 1


@dataclass(frozen=True)
class StoppingCriterion:
    eig_per_cost_threshold: float = 0.01
    consecutive_iterations: int = 3
    min_experiments: int = 20

    def __post_init__(self):
        assert self.consecutive_iterations >= 1


@dataclass(frozen=True)
class ProblemSpec:
    objectives: List[Objective]
    name: str = "unnamed_campaign"
    hard_constraints: List[HardConstraint] = field(default_factory=list)
    soft_constraints: List[SoftConstraint] = field(default_factory=list)
    budget: BudgetSpec = field(default_factory=BudgetSpec)
    stopping_criterion: StoppingCriterion = field(default_factory=StoppingCriterion)
    surrogate_type: str = "GP"
    candidate_pool_size: int = 100
    n_input_dims: int = 5
    random_seed: int = 42
    benchmark: str = "vlmop2"

    def __post_init__(self):
        assert len(self.objectives) >= 2, (
            "AEDE requires at least 2 objectives. "
            "Single-objective problems should use a simpler BO tool."
        )
        hard_props = {c.property for c in self.hard_constraints}
        soft_props = {c.property for c in self.soft_constraints}
        overlap = hard_props & soft_props
        assert not overlap, f"Properties {overlap} appear in both hard and soft constraints."
        assert self.budget.per_iteration_cap <= self.candidate_pool_size, (
            f"per_iteration_cap ({self.budget.per_iteration_cap}) cannot exceed "
            f"candidate_pool_size ({self.candidate_pool_size})"
        )

    def config_hash(self) -> str:
        data = asdict(self)
        canonical = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def reference_point(self):
        return np.zeros(len(self.objectives))

    def n_objectives(self) -> int:
        return len(self.objectives)

    def model_dump(self) -> dict:
        return asdict(self)

    def dict(self) -> dict:
        return asdict(self)


def load_spec_from_yaml(path: str) -> ProblemSpec:
    import yaml
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return load_spec_from_dict(data)


def load_spec_from_dict(data: dict) -> ProblemSpec:
    # Convert nested dicts to dataclasses
    objectives = [Objective(**o) if isinstance(o, dict) else o for o in data.get("objectives", [])]
    hard = [HardConstraint(**c) if isinstance(c, dict) else c for c in data.get("hard_constraints", [])]
    soft = [SoftConstraint(**c) if isinstance(c, dict) else c for c in data.get("soft_constraints", [])]
    budget = BudgetSpec(**data["budget"]) if isinstance(data.get("budget"), dict) else data.get("budget", BudgetSpec())
    stop = StoppingCriterion(**data["stopping_criterion"]) if isinstance(data.get("stopping_criterion"), dict) else data.get("stopping_criterion", StoppingCriterion())

    spec = ProblemSpec(
        name=data.get("name", "unnamed_campaign"),
        objectives=objectives,
        hard_constraints=hard,
        soft_constraints=soft,
        budget=budget,
        stopping_criterion=stop,
        surrogate_type=data.get("surrogate_type", "GP"),
        candidate_pool_size=data.get("candidate_pool_size", 100),
        n_input_dims=data.get("n_input_dims", 5),
        random_seed=data.get("random_seed", 42),
        benchmark=data.get("benchmark", "vlmop2"),
    )
    print(f"[ProblemFraming] Loaded spec '{spec.name}' — hash: {spec.config_hash()}")
    print(f"  Objectives : {[o.name for o in spec.objectives]}")
    print(f"  Budget     : {spec.budget.experiments} experiments")
    return spec
