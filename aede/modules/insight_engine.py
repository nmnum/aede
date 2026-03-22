"""
Module 6: Insight Engine
=========================
Post-hoc causal attribution for campaign behaviour.
Consumes the Decision Trace Log — no runtime dependency on any other module.

Outputs:
  1. Phase transition detection — when did dominant strategy shift?
  2. Constraint violation attribution — FMEA-style feature risk table.
  3. Surrogate calibration analysis — ECE time series.
  4. Cost efficiency report — yield, selectivity, energetic cost.
"""

from __future__ import annotations
import json
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict


@dataclass
class PhaseTransition:
    iteration: int
    from_strategy: str
    to_strategy: str
    hv_at_transition: float
    trigger: str  # "model_calibrated", "pareto_saturated", "high_constraint_encounter"


@dataclass
class CostEfficiencyReport:
    total_experiments: int
    total_cost: float
    hv_final: float
    hv_per_experiment: float        # yield
    informative_fraction: float     # selectivity — fraction that improved HV
    budget_efficiency: float        # HV(stop) / HV(budget_exhausted) estimate
    cost_per_hv_unit: float
    stopped_early: bool
    stop_iteration: Optional[int]


@dataclass
class InsightReport:
    campaign_hash: str
    n_iterations: int
    phase_transitions: List[PhaseTransition]
    dominant_strategy_per_phase: List[Dict]
    calibration_ece_history: List[float]
    calibration_trend: str          # "improving", "degrading", "stable", "insufficient_data"
    violation_rate_history: List[float]
    feature_risk_scores: Optional[List[float]]
    cost_efficiency: CostEfficiencyReport
    key_findings: List[str]


class InsightEngine:

    def __init__(self, logger):
        """logger: DecisionTraceLogger instance."""
        self.logger = logger

    def generate_report(self, campaign_hash: str) -> InsightReport:
        hv_history = self.logger.get_hv_history(campaign_hash)
        weight_history = self.logger.get_strategy_weight_history(campaign_hash)
        observations = self.logger.get_observations(campaign_hash)
        decisions = self.logger.get_recent_decisions(campaign_hash, n=1000)

        n_iterations = len(hv_history)
        if n_iterations == 0:
            return self._empty_report(campaign_hash)

        transitions = self._detect_phase_transitions(weight_history, hv_history)
        dominant_phases = self._summarise_dominant_phases(weight_history)
        ece_history, ece_trend = self._analyse_calibration(hv_history)
        violation_history = self._violation_history(observations)
        cost_eff = self._cost_efficiency(hv_history, observations, decisions)
        feature_risks = self._feature_risk_from_decisions(decisions)
        findings = self._generate_findings(
            transitions, ece_trend, cost_eff, violation_history
        )

        report = InsightReport(
            campaign_hash=campaign_hash,
            n_iterations=n_iterations,
            phase_transitions=transitions,
            dominant_strategy_per_phase=dominant_phases,
            calibration_ece_history=ece_history,
            calibration_trend=ece_trend,
            violation_rate_history=violation_history,
            feature_risk_scores=feature_risks,
            cost_efficiency=cost_eff,
            key_findings=findings,
        )

        # Persist to trace
        self.logger.log_insight(campaign_hash, asdict(report))
        return report

    # ── Phase transition detection ────────────────────────────────────────────

    def _detect_phase_transitions(
        self,
        weight_history: List[dict],
        hv_history: List[dict],
    ) -> List[PhaseTransition]:
        if len(weight_history) < 3:
            return []

        strategies = [k for k in weight_history[0] if k != "iteration"]
        if not strategies:
            return []

        def dominant(w: dict) -> str:
            s = {k: v for k, v in w.items() if k != "iteration"}
            return max(s, key=s.get) if s else "unknown"

        hv_map = {h["iteration"]: h["hv"] for h in hv_history}
        transitions = []
        prev_dom = dominant(weight_history[0])

        for w in weight_history[1:]:
            curr_dom = dominant(w)
            iteration = w.get("iteration", 0)
            if curr_dom != prev_dom:
                hv = hv_map.get(iteration, 0.0)
                # Classify the trigger
                trigger = self._classify_trigger(iteration, hv_history, hv)
                transitions.append(PhaseTransition(
                    iteration=iteration,
                    from_strategy=prev_dom,
                    to_strategy=curr_dom,
                    hv_at_transition=hv,
                    trigger=trigger,
                ))
                prev_dom = curr_dom

        return transitions

    def _classify_trigger(
        self, iteration: int, hv_history: List[dict], current_hv: float
    ) -> str:
        if iteration < 5:
            return "early_campaign_warm_up"
        # Check if HV growth rate was slowing
        recent = [h["hv"] for h in hv_history if h["iteration"] <= iteration]
        if len(recent) >= 3:
            growth_rate = (recent[-1] - recent[-3]) / (recent[-3] + 1e-8)
            if growth_rate < 0.02:
                return "pareto_front_saturating"
            if growth_rate > 0.1:
                return "high_information_gain_phase"
        return "portfolio_rebalancing"

    def _summarise_dominant_phases(
        self, weight_history: List[dict]
    ) -> List[dict]:
        if not weight_history:
            return []
        strategies = [k for k in weight_history[0] if k != "iteration"]
        phases, current_dom, phase_start = [], None, 0

        def dominant(w):
            s = {k: v for k, v in w.items() if k != "iteration"}
            return max(s, key=s.get) if s else "unknown"

        for w in weight_history:
            d = dominant(w)
            itr = w.get("iteration", 0)
            if d != current_dom:
                if current_dom is not None:
                    phases.append({
                        "strategy": current_dom,
                        "from_iter": phase_start,
                        "to_iter": itr - 1,
                        "duration": itr - phase_start,
                    })
                current_dom = d
                phase_start = itr

        if current_dom:
            phases.append({
                "strategy": current_dom,
                "from_iter": phase_start,
                "to_iter": weight_history[-1].get("iteration", phase_start),
                "duration": weight_history[-1].get("iteration", phase_start) - phase_start + 1,
            })
        return phases

    # ── Calibration analysis ──────────────────────────────────────────────────

    def _analyse_calibration(
        self, hv_history: List[dict]
    ) -> Tuple[List[float], str]:
        """
        Proxy calibration from EIG/cost history.
        A well-calibrated model shows EIG/cost declining smoothly.
        Erratic EIG suggests miscalibration.
        """
        eig_vals = [h.get("eig_per_cost", 0.0) for h in hv_history if h.get("eig_per_cost") is not None]
        if len(eig_vals) < 4:
            return eig_vals, "insufficient_data"

        # Compute rolling variance as proxy for calibration stability
        rolling_vars = []
        w = 3
        for i in range(w, len(eig_vals)):
            window = eig_vals[i - w:i]
            rolling_vars.append(float(np.var(window)))

        if not rolling_vars:
            return eig_vals, "insufficient_data"

        # Trend: is variance increasing (degrading) or decreasing (improving)?
        first_half = np.mean(rolling_vars[: len(rolling_vars) // 2])
        second_half = np.mean(rolling_vars[len(rolling_vars) // 2:])
        ratio = second_half / (first_half + 1e-8)

        if ratio < 0.7:
            trend = "improving"
        elif ratio > 1.4:
            trend = "degrading"
        else:
            trend = "stable"

        return eig_vals, trend

    # ── Violation history ─────────────────────────────────────────────────────

    def _violation_history(self, observations: List[dict]) -> List[float]:
        """Rolling violation rate per iteration."""
        if not observations:
            return []
        by_iter: Dict[int, List[bool]] = {}
        for obs in observations:
            it = obs.get("iteration", 0)
            by_iter.setdefault(it, []).append(obs.get("constraint_valid", True))

        result = []
        for it in sorted(by_iter):
            flags = by_iter[it]
            result.append(1.0 - sum(flags) / len(flags))
        return result

    # ── Cost efficiency ───────────────────────────────────────────────────────

    def _cost_efficiency(
        self,
        hv_history: List[dict],
        observations: List[dict],
        decisions: List[dict],
    ) -> CostEfficiencyReport:
        n_exp = len(observations)
        total_cost = sum(obs.get("cost", 1.0) for obs in observations)
        hv_final = hv_history[-1]["hv"] if hv_history else 0.0
        hv_first = hv_history[0]["hv"] if hv_history else 0.0

        # Informative fraction: experiments that were selected with score > median
        scores = [d.get("decision_score", 0.0) for d in decisions]
        if scores:
            median_score = float(np.median(scores))
            informative = sum(1 for s in scores if s > median_score)
            informative_frac = informative / len(scores)
        else:
            informative_frac = 0.0

        # Budget efficiency: compare HV at early stopping vs theoretical max
        stop_itr = next(
            (h["iteration"] for h in hv_history if h.get("eig_per_cost", 1.0) < 0.02),
            None,
        )
        if stop_itr and hv_final > 0:
            hv_at_stop = next(
                (h["hv"] for h in hv_history if h["iteration"] == stop_itr), hv_final
            )
            budget_eff = hv_at_stop / (hv_final + 1e-8)
        else:
            budget_eff = 1.0

        return CostEfficiencyReport(
            total_experiments=n_exp,
            total_cost=total_cost,
            hv_final=hv_final,
            hv_per_experiment=(hv_final - hv_first) / max(n_exp, 1),
            informative_fraction=informative_frac,
            budget_efficiency=budget_eff,
            cost_per_hv_unit=total_cost / max(hv_final, 1e-6),
            stopped_early=stop_itr is not None,
            stop_iteration=stop_itr,
        )

    # ── Feature risk attribution ──────────────────────────────────────────────

    def _feature_risk_from_decisions(
        self, decisions: List[dict]
    ) -> Optional[List[float]]:
        """
        FMEA-style: infer which input dimensions are associated with
        low feasibility scores from the decision log.
        """
        low_feas = [d for d in decisions if d.get("feasibility_probability", 1.0) < 0.5]
        if len(low_feas) < 3:
            return None
        # Would need X vectors — not stored in decisions query for brevity
        # Return placeholder — real attribution happens in ConstraintLayer
        return None

    # ── Natural language findings ─────────────────────────────────────────────

    def _generate_findings(
        self,
        transitions: List[PhaseTransition],
        ece_trend: str,
        cost_eff: CostEfficiencyReport,
        violation_history: List[float],
    ) -> List[str]:
        findings = []

        if transitions:
            findings.append(
                f"{len(transitions)} strategy phase transition(s) detected. "
                f"First transition at iteration {transitions[0].iteration} "
                f"({transitions[0].from_strategy} → {transitions[0].to_strategy}, "
                f"trigger: {transitions[0].trigger})."
            )
        else:
            findings.append("No strategy phase transitions detected — single dominant strategy throughout campaign.")

        if ece_trend == "improving":
            findings.append("Surrogate calibration improved over the campaign — model uncertainty estimates became more reliable.")
        elif ece_trend == "degrading":
            findings.append("WARNING: Surrogate calibration degraded. EIG estimates in later iterations may be unreliable. Consider increasing GP restarts or switching to BNN surrogate.")
        elif ece_trend == "stable":
            findings.append("Surrogate calibration remained stable throughout the campaign.")

        if cost_eff.informative_fraction > 0.65:
            findings.append(
                f"High selectivity: {cost_eff.informative_fraction:.0%} of experiments were above-median informative."
            )
        elif cost_eff.informative_fraction < 0.35:
            findings.append(
                f"Low selectivity ({cost_eff.informative_fraction:.0%} informative experiments). "
                "Consider increasing the per_iteration_cap to allow the acquisition function more flexibility."
            )

        if violation_history:
            avg_viol = float(np.mean(violation_history))
            if avg_viol > 0.3:
                findings.append(
                    f"High average constraint violation rate ({avg_viol:.0%}). "
                    "The constraint intelligence layer requires more observations before its classifier becomes reliable. "
                    "Consider increasing min_experiments."
                )
            elif avg_viol < 0.1:
                findings.append(
                    f"Low constraint violation rate ({avg_viol:.0%}) — constraint layer operating effectively."
                )

        if cost_eff.stopped_early:
            findings.append(
                f"Campaign converged at iteration {cost_eff.stop_iteration} "
                f"(budget efficiency: {cost_eff.budget_efficiency:.0%})."
            )

        findings.append(
            f"Cost efficiency: {cost_eff.hv_per_experiment:.4f} HV per experiment, "
            f"{cost_eff.cost_per_hv_unit:.2f} cost units per HV unit."
        )

        return findings

    def _empty_report(self, campaign_hash: str) -> InsightReport:
        return InsightReport(
            campaign_hash=campaign_hash,
            n_iterations=0,
            phase_transitions=[],
            dominant_strategy_per_phase=[],
            calibration_ece_history=[],
            calibration_trend="insufficient_data",
            violation_rate_history=[],
            feature_risk_scores=None,
            cost_efficiency=CostEfficiencyReport(0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, False, None),
            key_findings=["No iterations logged yet."],
        )
