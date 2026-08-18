import numpy as np
from floris import FlorisModel

# =====================================================================
# 1. ACOUSTIC ESTIMATION FUNCTIONS (ISO 9613-2)
# =====================================================================


def estimate_turbine_sound_power(wind_speed):
    """
    Estimates NREL 5MW sound power level L_w (dBA) based on incoming hub-height wind speed.
    Linear ramp between cut-in (3 m/s) and rated (11 m/s).
    """
    if wind_speed < 3.0:
        return 0.0
    elif wind_speed >= 11.0:
        return 106.0
    else:
        return 90.0 + ((wind_speed - 3.0) / (11.0 - 3.0)) * (106.0 - 90.0)


def estimate_farm_dBA(
    turbine_coords,
    receiver_coords,
    turbine_wind_speeds,
    hub_height=90.0,
    only_aggregate=True,
):
    """
    Estimates total dBA at receiver points using waked individual turbine wind speeds.

    Parameters:
    -----------
    turbine_coords       : np.ndarray, shape (N, 2)
                           (x, y) coordinates of N turbines.
    receiver_coords      : np.ndarray, shape (M, 2)
                           (x, y) coordinates of boundary receiver points.
    turbine_wind_speeds  : np.ndarray or list, shape (N,)
                           Effective incoming wind speed at each turbine (m/s).
    hub_height           : float
                           Hub height in meters (default 90m for NREL 5MW).

    Returns:
    --------
    l_total : np.ndarray, shape (M,)
              Total dBA at each receiver point.
    """
    turbine_coords = np.atleast_2d(turbine_coords)
    receiver_coords = np.atleast_2d(receiver_coords)
    turbine_wind_speeds = np.array(turbine_wind_speeds)

    n_turbines = len(turbine_coords)
    n_receivers = len(receiver_coords)

    # Calculate individual sound power output L_w for each turbine
    l_w_array = np.array([estimate_turbine_sound_power(v) for v in turbine_wind_speeds])

    # Calculate 3D slant distances from each turbine to each receiver
    # shape: (N_turbines, M_receivers)
    dx = turbine_coords[:, 0, np.newaxis] - receiver_coords[:, 0]
    dy = turbine_coords[:, 1, np.newaxis] - receiver_coords[:, 1]
    d_3d = np.sqrt(dx**2 + dy**2 + hub_height**2)
    d_3d = np.maximum(d_3d, 1.0)  # Avoid division by zero

    # Sound pressure level L_p per turbine at each receiver point (ISO 9613-2)
    # L_p = L_w - 20*log10(d) - 8 - 0.005*d
    l_p = l_w_array[:, np.newaxis] - 20 * np.log10(d_3d) - 8.0 - 0.005 * d_3d

    aggregate = 10.0 * np.log10(np.sum(10.0 ** (l_p / 10.0), axis=0))

    if only_aggregate:
        return aggregate
    return {"aggregate": aggregate, "individuals": l_p}


# =====================================================================
# 2. FLORIS EVALUATION FUNCTION
# =====================================================================

from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_GCH = MODULE_DIR / "gch.yaml"


def evaluate_floris_farm(
    coords,
    wind_speeds=[10.0],
    wind_directions=[270.0],
    turbulence_intensities=[0.06],
    turbine_type="nrel_5mw",
    floris_model_type=DEFAULT_GCH,
):
    """
    Evaluates power output and effective rotor-averaged wind velocities using FLORIS.
    """

    coords = np.array(coords)
    n_turbines = coords.shape[0]

    # Handle string paths or Path objects safely
    config_path = Path(floris_model_type)
    if not config_path.is_absolute():
        config_path = (MODULE_DIR / config_path).resolve()

    if not config_path.exists():
        raise FileNotFoundError(
            f"FLORIS configuration file not found at: {config_path}"
        )

    # Initialize FLORIS
    fmodel = FlorisModel(str(config_path))

    # Assign inputs and explicitly pass reference_wind_height at the exact same time
    fmodel.set(
        layout_x=coords[:, 0],
        layout_y=coords[:, 1],
        wind_speeds=wind_speeds,
        wind_directions=wind_directions,
        turbulence_intensities=turbulence_intensities,
        turbine_type=[turbine_type] * n_turbines,
        # ADD THIS LINE: Explicitly match the reference height to your turbine's hub height
        reference_wind_height=-1,
    )

    fmodel.assign_hub_height_to_ref_height()

    # Solve steady-state flow field
    fmodel.run()

    # Extract power outputs (kW) and effective wind velocities (m/s)
    turbine_powers_kw = fmodel.get_turbine_powers().flatten()
    turbine_velocities = fmodel.turbine_average_velocities.flatten()
    total_power_kw = np.sum(turbine_powers_kw)

    return {
        "total_power_kw": total_power_kw / 1000,
        "turbine_powers_kw": turbine_powers_kw / 1000,
        "turbine_velocities": turbine_velocities,
        "floris_model": fmodel,
    }
