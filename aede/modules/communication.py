"""
Module 7a: Communication Layer — Rationale Generation
=======================================================
Translates Decision Engine manifests into human-readable rationales.
No external dependencies — pure string formatting.
"""

from __future__ import annotations
from typing import List
from .decision_engine import DecisionManifest, DecisionRecord


class CommunicationLayer:

    def annotate_manifest(self, manifest: DecisionManifest) -> DecisionManifest:
        """Attach human-readable rationales to all selected decisions."""
        for record in manifest.selected:
            record.rationale = self._build_rationale(record, manifest)
        return manifest

    def _build_rationale(
        self, record: DecisionRecord, manifest: DecisionManifest
    ) -> str:
        lines = []
        lines.append(f"Decision: Run candidate {record.candidate_id}")
        lines.append(
            f"Score: {record.decision_score:.4f}  |  "
            f"Rank: {record.rank}/{len(manifest.all_scored)}  |  "
            f"Cost: {record.experiment_cost:.2f} units"
        )
        lines.append("")

        # Why
        why_parts = []
        if record.expected_pareto_improvement > 0.01:
            why_parts.append(
                f"expected Pareto HV improvement of {record.expected_pareto_improvement:.4f}"
            )
        if record.uncertainty_discount > 0.6:
            why_parts.append("high surrogate uncertainty in this region (exploration value)")
        elif record.uncertainty_discount < 0.3:
            why_parts.append("low surrogate uncertainty (exploitation of known region)")
        if record.feasibility_probability > 0.8:
            why_parts.append(f"high feasibility confidence ({record.feasibility_probability:.0%})")
        if record.provenance == "diversity":
            why_parts.append("space-filling diversity selection")

        if why_parts:
            lines.append("Why: " + "; ".join(why_parts) + ".")
        else:
            lines.append("Why: highest cost-adjusted information gain in filtered pool.")

        # Risk
        if record.feasibility_probability < 0.7:
            lines.append(
                f"Risk: {(1 - record.feasibility_probability):.0%} soft constraint violation probability."
            )

        # Provenance
        strategy_desc = {
            "bo": "Bayesian optimisation (qNEHVI acquisition)",
            "nsga": "evolutionary search (NSGA-II)",
            "diversity": "diversity sampling (MaxMin distance)",
        }
        lines.append(f"Strategy: {strategy_desc.get(record.provenance, record.provenance)}.")

        if record.gittins_index is not None:
            lines.append(f"Gittins index: {record.gittins_index:.4f} (opportunity-cost adjusted).")

        return "\n".join(lines)

    def format_stop_notice(self, manifest: DecisionManifest) -> str:
        return (
            f"\n{'='*60}\n"
            f"CAMPAIGN STOPPING CRITERION MET (iteration {manifest.iteration})\n"
            f"{manifest.stop_reason}\n"
            f"Final Pareto hypervolume: {manifest.pareto_hypervolume:.4f}\n"
            f"Budget remaining: {manifest.budget_remaining} experiments\n"
            f"{'='*60}\n"
        )

    def format_iteration_summary(self, manifest: DecisionManifest) -> str:
        lines = [
            f"\n[Iteration {manifest.iteration}]",
            f"  HV: {manifest.pareto_hypervolume:.4f}  |  "
            f"EIG/cost: {manifest.eig_per_cost:.4f}  |  "
            f"Budget left: {manifest.budget_remaining}",
        ]
        if manifest.strategy_weights:
            w_str = "  ".join(
                f"{k}: {v:.2f}" for k, v in manifest.strategy_weights.items()
            )
            lines.append(f"  Weights: {w_str}")
        lines.append(f"  Selected: {len(manifest.selected)} candidates")
        return "\n".join(lines)
