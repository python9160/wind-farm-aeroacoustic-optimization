"""Vectorized machine-learning acoustic surrogate.

Supports both legacy single-layout inputs and batched layouts/conditions.

Expected saved bundle format:
{
    "model": fitted_sklearn_estimator,
    "feature_columns": ["v", "log10_r", "cos_theta", "sin_theta"],
    ...
}
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

import joblib
import numpy as np
import pandas as pd


class AASurrogateMLVec:
    """Predict and aggregate wind-farm noise with a trained sklearn model.

    Accepted geometry shapes
    ------------------------
    turbine_coords:
        ``(N, 2)`` for one layout, or ``(B, N, 2)`` for B layouts.

    receiver_coords:
        ``(M, 2)`` shared by every layout, or ``(B, M, 2)``.

    turbine_wind_speeds:
        scalar, ``(N,)``, ``(B,)``, or ``(B, N)``.

    wind_dir_deg:
        scalar or ``(B,)``.

    Return shapes
    -------------
    For a single layout, legacy shapes are preserved:
        aggregate ``(M,)``; individuals ``(N, M)``.

    For B layouts:
        aggregate ``(B, M)``; individuals ``(B, N, M)``.
    """

    REQUIRED_FEATURES = ("v", "log10_r", "cos_theta", "sin_theta")

    def __init__(
        self,
        model_path: str | Path = "dba_model_outputs/best_dba_model.joblib",
        r_ref: float = 200.0,
        minimum_distance: float = 1.0e-6,
        prediction_batch_size: int | None = 250_000,
    ) -> None:
        self.model_path = Path(model_path)
        self.r_ref = float(r_ref)  # Backward compatibility; intentionally unused.
        self.minimum_distance = float(minimum_distance)
        self.prediction_batch_size = prediction_batch_size

        if self.minimum_distance <= 0.0:
            raise ValueError("minimum_distance must be greater than zero.")
        if prediction_batch_size is not None and prediction_batch_size <= 0:
            raise ValueError("prediction_batch_size must be positive or None.")
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        loaded: Any = joblib.load(self.model_path)
        if isinstance(loaded, dict) and "model" in loaded:
            self.model = loaded["model"]
            self.feature_columns = list(
                loaded.get("feature_columns", self.REQUIRED_FEATURES)
            )
            self.model_name = loaded.get("model_name", type(self.model).__name__)
            self.feature_set_name = loaded.get("feature_set_name")
            self.metadata = {k: v for k, v in loaded.items() if k != "model"}
        else:
            self.model = loaded
            self.feature_columns = list(self.REQUIRED_FEATURES)
            self.model_name = type(self.model).__name__
            self.feature_set_name = None
            self.metadata = {}

        if not hasattr(self.model, "predict"):
            raise TypeError("The loaded object does not provide predict().")
        if tuple(self.feature_columns) != self.REQUIRED_FEATURES:
            raise ValueError(
                "Expected feature order "
                f"{list(self.REQUIRED_FEATURES)}, got {self.feature_columns}."
            )

    @staticmethod
    def _as_coordinate_batch(name: str, values: Any) -> tuple[np.ndarray, bool]:
        """Return coordinates as (B, P, 2) and whether input was unbatched."""
        arr = np.asarray(values, dtype=np.float64)
        was_unbatched = arr.ndim <= 2

        if arr.ndim == 1:
            if arr.size != 2:
                raise ValueError(f"{name} 1D input must be [x, y].")
            arr = arr.reshape(1, 1, 2)
        elif arr.ndim == 2:
            if arr.shape[1] != 2:
                raise ValueError(f"{name} must end in dimension 2; got {arr.shape}.")
            arr = arr[None, :, :]
        elif arr.ndim == 3:
            if arr.shape[2] != 2:
                raise ValueError(f"{name} must end in dimension 2; got {arr.shape}.")
        else:
            raise ValueError(
                f"{name} must have shape (P,2) or (B,P,2); got {arr.shape}."
            )

        if arr.shape[1] == 0:
            raise ValueError(f"{name} cannot be empty.")
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} contains NaN or infinite values.")
        return arr, was_unbatched

    @staticmethod
    def _broadcast_batch_axis(
        arr: np.ndarray, batch_size: int, name: str
    ) -> np.ndarray:
        if arr.shape[0] == batch_size:
            return arr
        if arr.shape[0] == 1:
            return np.broadcast_to(arr, (batch_size, *arr.shape[1:]))
        raise ValueError(
            f"{name} has batch size {arr.shape[0]}, expected 1 or {batch_size}."
        )

    @staticmethod
    def _prepare_wind_speeds(
        values: Any, batch_size: int, n_turbines: int
    ) -> np.ndarray:
        """Broadcast wind speed to shape (B, N)."""
        arr = np.asarray(values, dtype=np.float64)

        if arr.ndim == 0:
            out = np.full((batch_size, n_turbines), arr.item())
        elif arr.ndim == 1:
            if arr.size == 1:
                out = np.full((batch_size, n_turbines), arr.item())
            elif arr.size == n_turbines:
                out = np.broadcast_to(arr[None, :], (batch_size, n_turbines))
            elif arr.size == batch_size:
                out = np.broadcast_to(arr[:, None], (batch_size, n_turbines))
            else:
                raise ValueError(
                    "1D turbine_wind_speeds must have length 1, N turbines, "
                    f"or B layouts; got {arr.size}, N={n_turbines}, B={batch_size}."
                )
        elif arr.ndim == 2:
            try:
                out = np.broadcast_to(arr, (batch_size, n_turbines))
            except ValueError as exc:
                raise ValueError(
                    "2D turbine_wind_speeds must broadcast to (B,N); "
                    f"got {arr.shape}, target {(batch_size, n_turbines)}."
                ) from exc
        else:
            raise ValueError("turbine_wind_speeds must be scalar, 1D, or 2D.")

        if not np.isfinite(out).all():
            raise ValueError("turbine_wind_speeds contains NaN or infinite values.")
        return np.asarray(out, dtype=np.float64)

    @staticmethod
    def _prepare_wind_directions(values: Any, batch_size: int) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim == 0:
            out = np.full(batch_size, arr.item())
        else:
            arr = arr.reshape(-1)
            if arr.size == 1:
                out = np.full(batch_size, arr.item())
            elif arr.size == batch_size:
                out = arr
            else:
                raise ValueError(
                    f"wind_dir_deg must be scalar or length B={batch_size}; got {arr.size}."
                )
        if not np.isfinite(out).all():
            raise ValueError("wind_dir_deg contains NaN or infinite values.")
        return np.mod(out, 360.0)

    def _predict_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Predict an (R,4) matrix, chunking only to control memory."""
        n_rows = matrix.shape[0]
        batch = self.prediction_batch_size

        def frame(block: np.ndarray) -> pd.DataFrame:
            return pd.DataFrame(block, columns=self.feature_columns)

        if batch is None or n_rows <= batch:
            return np.asarray(self.model.predict(frame(matrix)), dtype=np.float64)

        result = np.empty(n_rows, dtype=np.float64)
        for start in range(0, n_rows, batch):
            stop = min(start + batch, n_rows)
            result[start:stop] = self.model.predict(frame(matrix[start:stop]))
        return result

    @staticmethod
    def _energetic_sum_db(individual_dba: np.ndarray) -> np.ndarray:
        """Stable energetic sum over turbine axis (-2)."""
        peak = np.max(individual_dba, axis=-2, keepdims=True)
        relative = np.power(10.0, (individual_dba - peak) / 10.0)
        return np.squeeze(peak, axis=-2) + 10.0 * np.log10(np.sum(relative, axis=-2))

    def evaluate_farm_dBA(
        self,
        turbine_coords,
        receiver_coords,
        turbine_wind_speeds,
        wind_dir_deg: float | np.ndarray = 270.0,
        only_aggregate: bool = True,
        absorption: float = 0.005,
    ):
        """Predict dBA for one or many layouts in one vectorized call."""
        turbines, turbines_unbatched = self._as_coordinate_batch(
            "turbine_coords", turbine_coords
        )
        receivers, receivers_unbatched = self._as_coordinate_batch(
            "receiver_coords", receiver_coords
        )

        batch_size = max(turbines.shape[0], receivers.shape[0])
        turbines = self._broadcast_batch_axis(turbines, batch_size, "turbine_coords")
        receivers = self._broadcast_batch_axis(receivers, batch_size, "receiver_coords")

        n_turbines = turbines.shape[1]
        n_receivers = receivers.shape[1]
        wind_speeds = self._prepare_wind_speeds(
            turbine_wind_speeds, batch_size, n_turbines
        )
        wind_directions = self._prepare_wind_directions(wind_dir_deg, batch_size)

        if absorption != 0.005:
            warnings.warn(
                "absorption is ignored because attenuation is represented by log10_r.",
                RuntimeWarning,
                stacklevel=2,
            )

        # Shapes throughout: (B, N, M).
        dx = receivers[:, None, :, 0] - turbines[:, :, None, 0]
        dy = receivers[:, None, :, 1] - turbines[:, :, None, 1]
        distance = np.hypot(dx, dy)
        safe_r = np.maximum(distance, self.minimum_distance)

        # FLORIS meteorological direction -> Cartesian downwind heading.
        heading = np.deg2rad((270.0 - wind_directions) % 360.0)
        heading_cos = np.cos(heading)[:, None, None]
        heading_sin = np.sin(heading)[:, None, None]

        unit_x = dx / safe_r
        unit_y = dy / safe_r
        cos_theta = unit_x * heading_cos + unit_y * heading_sin
        sin_theta = unit_y * heading_cos - unit_x * heading_sin
        v = np.broadcast_to(
            wind_speeds[:, :, None], (batch_size, n_turbines, n_receivers)
        )

        # Build all B*N*M feature rows once and call sklearn in large blocks.
        feature_matrix = np.column_stack(
            (
                v.ravel(),
                np.log10(safe_r).ravel(),
                cos_theta.ravel(),
                sin_theta.ravel(),
            )
        )
        individuals = self._predict_matrix(feature_matrix).reshape(
            batch_size, n_turbines, n_receivers
        )
        if not np.isfinite(individuals).all():
            raise RuntimeError("The model returned NaN or infinite predictions.")

        aggregate = self._energetic_sum_db(individuals)

        # Preserve the original API for any effectively single-layout call.
        single_layout = batch_size == 1 and turbines_unbatched and receivers_unbatched
        if single_layout:
            aggregate_out = aggregate[0]
            individuals_out = individuals[0]
        else:
            aggregate_out = aggregate
            individuals_out = individuals

        if only_aggregate:
            return aggregate_out
        return {"aggregate": aggregate_out, "individuals": individuals_out}

    def predict_pairs(self, v, r, cos_theta, sin_theta) -> np.ndarray:
        """Vectorized prediction for already prepared features."""
        v_arr, r_arr, cos_arr, sin_arr = np.broadcast_arrays(
            np.asarray(v, dtype=np.float64),
            np.asarray(r, dtype=np.float64),
            np.asarray(cos_theta, dtype=np.float64),
            np.asarray(sin_theta, dtype=np.float64),
        )
        if not all(np.isfinite(x).all() for x in (v_arr, r_arr, cos_arr, sin_arr)):
            raise ValueError("Prediction inputs must be finite.")
        if np.any(r_arr <= 0.0):
            raise ValueError("All r values must be greater than zero.")

        shape = v_arr.shape
        matrix = np.column_stack(
            (
                v_arr.ravel(),
                np.log10(r_arr).ravel(),
                cos_arr.ravel(),
                sin_arr.ravel(),
            )
        )
        return self._predict_matrix(matrix).reshape(shape)
