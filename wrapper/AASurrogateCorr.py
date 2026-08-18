import json
import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d


class AASurrogateCorr:
    def __init__(self, json_path="noise_lookup_table.json", r_ref=200.0):
        """
        Loads the OpenFAST 32-observer noise table and initializes a fast 2D interpolator.
        Also loads an optional directional 'correction' offset array.

        Parameters:
        -----------
        json_path : str
            Path to the JSON file generated from OpenFAST runs.
        r_ref : float
            Reference radius at which observers were placed in OpenFAST (default 200m).
        """
        self.r_ref = r_ref

        # Load JSON dataset
        with open(json_path, "r") as f:
            payload = json.load(f)

        data = payload["data"]

        # Extract and sort unique wind speeds
        self.wind_speeds = np.array(
            sorted([float(ws) for ws in data.keys()]), dtype=np.float64
        )

        # Build 32 angles spanning 0 to 2pi radians
        num_observers = payload["metadata"].get("num_observers", 32)
        self.angles_rad = np.linspace(0, 2 * np.pi, num_observers, endpoint=False)

        # Build 2D lookup matrix: shape (len(wind_speeds), num_observers)
        oaspl_matrix = np.zeros(
            (len(self.wind_speeds), num_observers), dtype=np.float64
        )
        for i, ws in enumerate(self.wind_speeds):
            str_key = str(ws) if str(ws) in data else f"{ws:.1f}"
            oaspl_matrix[i, :] = data[str_key]

        # Initialize fast 2D interpolator over (Wind Speed, Azimuth Angle)
        self.interpolator = RegularGridInterpolator(
            (self.wind_speeds, self.angles_rad),
            oaspl_matrix,
            bounds_error=False,
            fill_value=None,
        )

        # Initialize directional correction interpolator if data exists
        self.correction_func = None
        if "correction" in payload:
            correction_data = np.array(payload["correction"], dtype=np.float64)
            n_corr = len(correction_data)

            # Create angles for the correction array.
            # To handle 360-degree wrap-around smoothly, append the 0-deg value to the end (at 2*pi).
            corr_angles = np.linspace(0, 2 * np.pi, n_corr + 1, endpoint=True)
            corr_values = np.append(correction_data, correction_data[0])

            self.correction_func = interp1d(
                corr_angles,
                corr_values,
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate",
            )

    def evaluate_farm_dBA(
        self,
        turbine_coords,
        receiver_coords,
        turbine_wind_speeds,
        wind_dir_deg=270.0,
        only_aggregate=True,
    ):
        """
        Direct drop-in replacement for estimate_farm_dBA.
        Vectorized evaluation of noise propagation across M boundary receivers.

        Parameters:
        -----------
        turbine_coords      : np.ndarray, shape (N, 2)
            (x, y) coordinates of N turbines.
        receiver_coords     : np.ndarray, shape (M, 2)
            (x, y) coordinates of M boundary receiver points.
        turbine_wind_speeds : np.ndarray or list, shape (N,)
            Waked effective wind speed per turbine from FLORIS (m/s).
        wind_dir_deg        : float, default 270.0
            Dominant incoming wind direction in degrees.

        Returns:
        --------
        l_total             : np.ndarray, shape (M,)
            Total combined dBA at each of the M receiver points.
        """
        turbine_coords = np.atleast_2d(turbine_coords)  # Shape: (N, 2)
        receiver_coords = np.atleast_2d(receiver_coords)  # Shape: (M, 2)
        turbine_wind_speeds = np.asarray(
            turbine_wind_speeds, dtype=np.float64
        )  # Shape: (N,)

        n_turbines = len(turbine_coords)
        n_receivers = len(receiver_coords)
        wind_dir_rad = np.radians(wind_dir_deg)

        # 1. Coordinate differences matrix -> Shape: (N, M)
        dx = receiver_coords[:, 0] - turbine_coords[:, 0, np.newaxis]
        dy = receiver_coords[:, 1] - turbine_coords[:, 1, np.newaxis]

        # 2. Slant distance calculations using constant 88m height offset (88^2 = 7744.0)
        d_sq = dx**2 + dy**2 + 7744.0
        d_3d = np.sqrt(d_sq)  # Needed for atmospheric absorption step

        # 3. Relative azimuth angle from each turbine to each receiver -> Shape: (N, M)
        angles_to_rec = np.mod(np.arctan2(dy, dx) - wind_dir_rad, 2 * np.pi)

        # 4. Prepare evaluation grid for 2D interpolator -> Shape: (N * M, 2)
        ws_grid = np.repeat(turbine_wind_speeds[:, np.newaxis], n_receivers, axis=1)
        eval_points = np.column_stack((ws_grid.ravel(), angles_to_rec.ravel()))

        # 5. Interpolate base sound level at reference distance (R_ref = 200m) -> Shape: (N, M)
        l_ref = self.interpolator(eval_points).reshape((n_turbines, n_receivers))

        # 6. Inverse-square distance scaling + atmospheric absorption (-0.005 dBA/m)
        # Equivalent to: l_ref - 20*log10(d_3d / R_ref) - 0.005*(d_3d - R_ref)
        l_p = (
            l_ref
            - 10.0 * np.log10(d_sq / (self.r_ref**2))
            - 0.005 * (d_3d - self.r_ref)
        )

        # 7. Apply directional correction offsets if present in the JSON
        if self.correction_func is not None:
            # angles_to_rec is already wrapped to [0, 2*pi), safe to pass directly
            interpolated_offsets = self.correction_func(angles_to_rec)
            l_p = l_p - interpolated_offsets

        # 8. Aggregate sound pressure levels
        aggregate = 10.0 * np.log10(np.sum(10.0 ** (l_p / 10.0), axis=0))

        if only_aggregate:
            return aggregate
        return {"aggregate": aggregate, "individuals": l_p}
