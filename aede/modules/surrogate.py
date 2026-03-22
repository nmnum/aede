"""
Surrogate Model Layer (sklearn backend)
=========================================
Gaussian Process surrogate for multi-objective optimisation.
One GP per objective — independent models, joint uncertainty.

This is the default (lightweight) backend.
The production BoTorch backend is in aede/modules/surrogate_botorch.py
and is a drop-in replacement that respects the same interface.

Interface contract:
    fit(X, Y)
    predict(X) -> (mu, sigma)  shapes: (n, n_obj), (n, n_obj)
    calibration_error(X, Y_true) -> float (ECE)
"""

from __future__ import annotations
import numpy as np
from typing import Optional, List, Tuple
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler


class GPSurrogate:
    """
    Independent GP per objective with Matérn-5/2 kernel.
    Inputs are standardised internally.
    Outputs are in the original (maximisation) scale.
    """

    def __init__(
        self,
        n_objectives: int,
        n_restarts: int = 3,
        noise_level: float = 0.01,
        seed: Optional[int] = None,
    ):
        self.n_objectives = n_objectives
        self.n_restarts = n_restarts
        self.seed = seed
        self._is_fit = False
        self._X_scaler = StandardScaler()
        self._Y_scalers = [StandardScaler() for _ in range(n_objectives)]

        kernel = Matern(nu=2.5, length_scale_bounds=(1e-3, 10.0)) + WhiteKernel(
            noise_level=noise_level, noise_level_bounds=(1e-5, 1.0)
        )
        self._models: List[GaussianProcessRegressor] = [
            GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=n_restarts,
                normalize_y=False,
                random_state=seed,
            )
            for _ in range(n_objectives)
        ]
        self._X_train: Optional[np.ndarray] = None
        self._Y_train: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "GPSurrogate":
        """
        Fit all GP models.
        X: (n_samples, n_features)
        Y: (n_samples, n_objectives)  — maximisation convention
        """
        assert Y.shape[1] == self.n_objectives, (
            f"Expected {self.n_objectives} objective columns, got {Y.shape[1]}"
        )
        Xs = self._X_scaler.fit_transform(X)
        self._X_train = Xs
        self._Y_train = Y.copy()

        for i, (model, scaler) in enumerate(zip(self._models, self._Y_scalers)):
            yi = scaler.fit_transform(Y[:, i].reshape(-1, 1)).flatten()
            model.fit(Xs, yi)

        self._is_fit = True
        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (mu, sigma) both in original objective scale.
        Shapes: (n, n_obj), (n, n_obj)
        """
        assert self._is_fit, "Surrogate must be fit before prediction"
        Xs = self._X_scaler.transform(X)
        mus, sigmas = [], []
        for model, scaler in zip(self._models, self._Y_scalers):
            mu_s, sigma_s = model.predict(Xs, return_std=True)
            # Inverse transform mean
            mu_orig = scaler.inverse_transform(mu_s.reshape(-1, 1)).flatten()
            # Scale sigma back (no shift — std scales by scale_ only)
            sigma_orig = sigma_s * scaler.scale_[0]
            mus.append(mu_orig)
            sigmas.append(sigma_orig)

        return np.column_stack(mus), np.column_stack(sigmas)

    def predict_fn(self, X: np.ndarray):
        """Callable compatible with IVO.register_surrogate()."""
        return self.predict(X)

    def sample_posterior(self, X: np.ndarray, n_samples: int = 256) -> np.ndarray:
        """
        Draw n_samples from the joint posterior at X.
        Returns: (n_samples, n_points, n_objectives)
        """
        assert self._is_fit
        Xs = self._X_scaler.transform(X)
        all_samples = []
        for model, scaler in zip(self._models, self._Y_scalers):
            y_s = model.sample_y(Xs, n_samples=n_samples, random_state=self.seed)
            # y_s shape: (n_points, n_samples)
            y_orig = scaler.inverse_transform(y_s).T  # (n_samples, n_points)
            all_samples.append(y_orig)
        # Stack: (n_samples, n_points, n_objectives)
        return np.stack(all_samples, axis=-1)

    def expected_calibration_error(
        self, X_test: np.ndarray, Y_true: np.ndarray, n_bins: int = 10
    ) -> float:
        """
        Expected Calibration Error (ECE) — fraction of true values falling
        outside their predicted credible intervals vs. expected frequency.
        Averaged across objectives.
        """
        if not self._is_fit or len(X_test) < 5:
            return float("nan")
        mu, sigma = self.predict(X_test)
        eces = []
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        for obj_i in range(self.n_objectives):
            errors = np.abs(Y_true[:, obj_i] - mu[:, obj_i])
            z_scores = errors / np.maximum(sigma[:, obj_i], 1e-8)
            # For a calibrated GP: fraction within z*sigma should match normal CDF
            from scipy.stats import norm
            ece_obj = 0.0
            for j in range(n_bins):
                lo, hi = bins[j], bins[j + 1]
                p_mid = (lo + hi) / 2.0
                z = norm.ppf((1 + p_mid) / 2)
                expected_frac = p_mid
                actual_frac = float(np.mean(z_scores <= z))
                ece_obj += abs(actual_frac - expected_frac) / n_bins
            eces.append(ece_obj)
        return float(np.mean(eces))

    @property
    def n_observations(self) -> int:
        return len(self._X_train) if self._X_train is not None else 0

    def state_snapshot(self) -> dict:
        """Lightweight snapshot for the Decision Trace Logger."""
        if not self._is_fit:
            return {"fit": False}
        mu, sigma = self.predict(self._X_train)
        return {
            "fit": True,
            "n_observations": self.n_observations,
            "mean_uncertainty": float(np.mean(sigma)),
            "kernel_params": [
                str(m.kernel_) for m in self._models
            ],
        }
