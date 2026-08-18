"""Machine-learning replacement for the lookup-table AASurrogate.

Expected saved bundle format (from the training notebook):
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


class AASurrogateML:
    """Predict and aggregate wind-farm noise using a trained scikit-learn model.

    This class is designed as a direct replacement for the previous JSON lookup-table
    implementation. The public ``evaluate_farm_dBA`` method keeps the same arguments
    and return structure.

    The trained model is expected to predict the received dBA for one turbine-receiver
    pair using these features:

    - ``v``: turbine wind speed
    - ``log10_r``: base-10 logarithm of turbine-receiver horizontal distance
    - ``cos_theta``: cosine of receiver direction relative to downwind
    - ``sin_theta``: sine of receiver direction relative to downwind

    Because distance is already an input to the model, this class does not apply an
    additional inverse-square spreading correction after prediction.
    """

    REQUIRED_FEATURES = ("v", "log10_r", "cos_theta", "sin_theta")

    def __init__(
        self,
        model_path: str | Path = "dba_model_outputs/best_dba_model.joblib",
        r_ref: float = 200.0,
        minimum_distance: float = 1.0e-6,
        prediction_batch_size: int | None = 250_000,
    ) -> None:
        """Load the trained model bundle.

        Parameters
        ----------
        model_path:
            Path to ``best_dba_model.joblib`` created by the training notebook.
        r_ref:
            Retained only for backward compatibility with the old constructor.
            It is not used by the ML model because ``log10_r`` is predicted directly.
        minimum_distance:
            Small positive value used to protect ``log10(r)`` when a receiver lies
            exactly on a turbine coordinate.
        prediction_batch_size:
            Maximum number of turbine-receiver rows predicted in one call. Set to
            ``None`` to predict all rows at once.
        """
        self.model_path = Path(model_path)
        self.r_ref = float(
            r_ref
        )  # Backward-compatible attribute; intentionally unused.
        self.minimum_distance = float(minimum_distance)
        self.prediction_batch_size = prediction_batch_size

        if self.minimum_distance <= 0:
            raise ValueError("minimum_distance must be greater than zero.")
        if prediction_batch_size is not None and prediction_batch_size <= 0:
            raise ValueError("prediction_batch_size must be positive or None.")
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        loaded: Any = joblib.load(self.model_path)

        if isinstance(loaded, dict) and "model" in loaded:
            self.model = loaded["model"]
            self.feature_columns = list(
                loaded.get("feature_columns", self.REQUIRED_FEATURES)
            )
            self.model_name = loaded.get("model_name", type(self.model).__name__)
            self.feature_set_name = loaded.get("feature_set_name")
            self.metadata = {
                key: value for key, value in loaded.items() if key != "model"
            }
        else:
            # Also support a joblib file containing only the fitted estimator.
            self.model = loaded
            self.feature_columns = list(self.REQUIRED_FEATURES)
            self.model_name = type(self.model).__name__
            self.feature_set_name = None
            self.metadata = {}

        if not hasattr(self.model, "predict"):
            raise TypeError("The loaded object does not provide a predict() method.")

        missing = [
            feature
            for feature in self.REQUIRED_FEATURES
            if feature not in self.feature_columns
        ]
        unexpected = [
            feature
            for feature in self.feature_columns
            if feature not in self.REQUIRED_FEATURES
        ]
        if missing or unexpected:
            raise ValueError(
                "This replacement expects exactly the trained features "
                f"{list(self.REQUIRED_FEATURES)}. Bundle contains "
                f"{self.feature_columns}. Missing={missing}; unexpected={unexpected}."
            )

    @staticmethod
    def _validate_coordinates(name: str, values: Any) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        array = np.atleast_2d(array)

        if array.ndim != 2 or array.shape[1] != 2:
            raise ValueError(f"{name} must have shape (n, 2); received {array.shape}.")
        if array.shape[0] == 0:
            raise ValueError(f"{name} cannot be empty.")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains NaN or infinite values.")
        return array

    @staticmethod
    def _prepare_wind_speeds(values: Any, n_turbines: int) -> np.ndarray:
        speeds = np.asarray(values, dtype=np.float64).reshape(-1)

        # Accept a scalar wind speed and broadcast it to all turbines.
        if speeds.size == 1 and n_turbines > 1:
            speeds = np.full(n_turbines, speeds.item(), dtype=np.float64)

        if speeds.size != n_turbines:
            raise ValueError(
                "turbine_wind_speeds must contain one value per turbine; "
                f"received {speeds.size} values for {n_turbines} turbines."
            )
        if not np.isfinite(speeds).all():
            raise ValueError("turbine_wind_speeds contains NaN or infinite values.")
        return speeds

    def _predict_feature_frame(self, features: pd.DataFrame) -> np.ndarray:
        """Predict in batches to avoid excessive temporary memory use."""
        n_rows = len(features)
        batch_size = self.prediction_batch_size

        if batch_size is None or n_rows <= batch_size:
            return np.asarray(
                self.model.predict(features[self.feature_columns]), dtype=np.float64
            )

        predictions = np.empty(n_rows, dtype=np.float64)
        for start in range(0, n_rows, batch_size):
            stop = min(start + batch_size, n_rows)
            predictions[start:stop] = self.model.predict(
                features.iloc[start:stop][self.feature_columns]
            )
        return predictions

    def evaluate_farm_dBA(
        self,
        turbine_coords,
        receiver_coords,
        turbine_wind_speeds,
        wind_dir_deg: float = 270.0,
        only_aggregate: bool = True,
        absorption: float = 0.005,
    ):
        """Predict turbine noise and energetically aggregate it at each receiver.

        Parameters
        ----------
        turbine_coords : array-like, shape (N, 2)
            Turbine ``(x, y)`` coordinates.
        receiver_coords : array-like, shape (M, 2)
            Receiver ``(x, y)`` coordinates.
        turbine_wind_speeds : array-like, shape (N,), or scalar
            Effective wind speed for each turbine.
        wind_dir_deg : float, default 270.0
            FLORIS meteorological wind direction. The conversion matches the old
            implementation: ``heading = radians((270 - wind_dir_deg) % 360)``.
        only_aggregate : bool, default True
            Return only total dBA when True. When False, return a dictionary containing
            both aggregate and individual turbine contributions.
        absorption : float, default 0.005
            Retained for API compatibility. It is intentionally not applied because the
            trained model already uses distance to predict received dBA. Applying it here
            would generally double-count attenuation.

        Returns
        -------
        numpy.ndarray, shape (M,)
            Aggregate dBA at each receiver when ``only_aggregate=True``.
        dict
            ``{"aggregate": aggregate, "individuals": individual_dba}`` otherwise,
            where ``individual_dba`` has shape ``(N, M)``.
        """
        turbine_coords = self._validate_coordinates("turbine_coords", turbine_coords)
        receiver_coords = self._validate_coordinates("receiver_coords", receiver_coords)

        n_turbines = turbine_coords.shape[0]
        n_receivers = receiver_coords.shape[0]
        wind_speeds = self._prepare_wind_speeds(turbine_wind_speeds, n_turbines)

        if not np.isfinite(wind_dir_deg):
            raise ValueError("wind_dir_deg must be finite.")

        # Keep the parameter without using it so existing callers remain compatible.
        if absorption != 0.005:
            warnings.warn(
                "absorption is ignored by the ML surrogate because distance attenuation "
                "is already represented by log10_r in the trained model.",
                RuntimeWarning,
                stacklevel=2,
            )

        # Pairwise turbine-to-receiver displacement, each with shape (N, M).
        dx = receiver_coords[None, :, 0] - turbine_coords[:, None, 0]
        dy = receiver_coords[None, :, 1] - turbine_coords[:, None, 1]

        r = np.hypot(dx, dy)
        safe_r = np.maximum(r, self.minimum_distance)

        # Convert meteorological wind direction to the same Cartesian downwind heading
        # used by the previous implementation.
        heading = np.deg2rad((270.0 - float(wind_dir_deg)) % 360.0)
        heading_cos = np.cos(heading)
        heading_sin = np.sin(heading)

        # Unit vector from each turbine to each receiver.
        unit_x = dx / safe_r
        unit_y = dy / safe_r

        # theta = receiver angle - downwind heading.
        cos_theta = unit_x * heading_cos + unit_y * heading_sin
        sin_theta = unit_y * heading_cos - unit_x * heading_sin

        # Repeat each turbine's wind speed across all receivers.
        v = np.broadcast_to(wind_speeds[:, None], (n_turbines, n_receivers))

        feature_frame = pd.DataFrame(
            {
                "v": v.ravel(),
                "log10_r": np.log10(safe_r).ravel(),
                "cos_theta": cos_theta.ravel(),
                "sin_theta": sin_theta.ravel(),
            }
        )

        individual_dba = self._predict_feature_frame(feature_frame).reshape(
            n_turbines, n_receivers
        )

        if not np.isfinite(individual_dba).all():
            raise RuntimeError("The model returned NaN or infinite dBA predictions.")

        # Energetic summation in the linear intensity domain.
        aggregate = 10.0 * np.log10(
            np.sum(np.power(10.0, individual_dba / 10.0), axis=0)
        )

        if only_aggregate:
            return aggregate

        return {
            "aggregate": aggregate,
            "individuals": individual_dba,
        }

    def predict_pairs(
        self,
        v,
        r,
        cos_theta,
        sin_theta,
    ) -> np.ndarray:
        """Predict dBA for already-prepared turbine-receiver feature rows.

        This is useful for testing the model independently from farm geometry.
        Inputs follow NumPy broadcasting rules and the returned array has the resulting
        broadcast shape.
        """
        v_array, r_array, cos_array, sin_array = np.broadcast_arrays(
            np.asarray(v, dtype=np.float64),
            np.asarray(r, dtype=np.float64),
            np.asarray(cos_theta, dtype=np.float64),
            np.asarray(sin_theta, dtype=np.float64),
        )

        if not (
            np.isfinite(v_array).all()
            and np.isfinite(r_array).all()
            and np.isfinite(cos_array).all()
            and np.isfinite(sin_array).all()
        ):
            raise ValueError("Prediction inputs must all be finite.")
        if np.any(r_array <= 0):
            raise ValueError("All r values must be greater than zero.")

        original_shape = v_array.shape
        feature_frame = pd.DataFrame(
            {
                "v": v_array.ravel(),
                "log10_r": np.log10(r_array).ravel(),
                "cos_theta": cos_array.ravel(),
                "sin_theta": sin_array.ravel(),
            }
        )
        return self._predict_feature_frame(feature_frame).reshape(original_shape)
