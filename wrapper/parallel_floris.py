"""Parallel FLORIS evaluation for many layouts and many wind scenarios.

Place this file inside your ``wrapper`` package as ``parallel_floris.py`` or
copy the functions into your existing module. Worker functions must stay at
module scope for Windows multiprocessing.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
from floris import FlorisModel
from tqdm.auto import tqdm

_WORKER_FMODEL: Optional[FlorisModel] = None
_WORKER_TURBINE_TYPE: Optional[str] = None


def _initialize_floris_worker(config_path: str, turbine_type: str) -> None:
    """Create one reusable FLORIS model in each worker process."""
    global _WORKER_FMODEL, _WORKER_TURBINE_TYPE
    _WORKER_FMODEL = FlorisModel(config_path)
    _WORKER_TURBINE_TYPE = turbine_type


def _evaluate_layout_worker(task):
    """Evaluate all wind scenarios for one candidate layout."""
    global _WORKER_FMODEL, _WORKER_TURBINE_TYPE

    if _WORKER_FMODEL is None or _WORKER_TURBINE_TYPE is None:
        raise RuntimeError("FLORIS worker was not initialized.")

    (
        layout_index,
        layout,
        wind_speeds,
        wind_directions,
        turbulence_intensities,
    ) = task

    layout = np.asarray(layout, dtype=np.float64)
    wind_speeds = np.asarray(wind_speeds, dtype=np.float64).reshape(-1)
    wind_directions = np.asarray(wind_directions, dtype=np.float64).reshape(-1)
    turbulence_intensities = np.asarray(
        turbulence_intensities, dtype=np.float64
    ).reshape(-1)

    if layout.ndim != 2 or layout.shape[1] != 2:
        raise ValueError(
            f"Layout {layout_index} must have shape (n_turbines, 2); "
            f"received {layout.shape}."
        )

    n_scenarios = wind_speeds.size
    n_turbines = layout.shape[0]

    if wind_directions.size != n_scenarios:
        raise ValueError("wind_speeds and wind_directions must have equal lengths.")
    if turbulence_intensities.size != n_scenarios:
        raise ValueError(
            "turbulence_intensities must contain one value per wind scenario."
        )

    _WORKER_FMODEL.set(
        layout_x=layout[:, 0],
        layout_y=layout[:, 1],
        wind_speeds=wind_speeds,
        wind_directions=wind_directions,
        turbulence_intensities=turbulence_intensities,
        turbine_type=[_WORKER_TURBINE_TYPE] * n_turbines,
        reference_wind_height=-1,
    )
    _WORKER_FMODEL.assign_hub_height_to_ref_height()
    _WORKER_FMODEL.run()

    turbine_powers_w = np.asarray(
        _WORKER_FMODEL.get_turbine_powers(), dtype=np.float64
    ).reshape(n_scenarios, n_turbines)
    turbine_velocities = np.asarray(
        _WORKER_FMODEL.turbine_average_velocities, dtype=np.float64
    ).reshape(n_scenarios, n_turbines)

    scenario_total_powers_kw = turbine_powers_w.sum(axis=1) / 1_000.0

    return layout_index, scenario_total_powers_kw, turbine_velocities


def evaluate_population_floris_parallel(
    population,
    *,
    wind_speeds,
    wind_directions,
    turbulence_intensities,
    config_path,
    turbine_type: str = "nrel_5mw",
    max_workers: int = 6,
    chunksize: int = 1,
    show_progress: bool = True,
):
    """Evaluate P layouts under S paired wind scenarios.

    Returns
    -------
    dict
        ``powers_kw`` has shape ``(P, S)``.
        ``waked_velocities`` has shape ``(P, S, N)``.
    """
    population = np.asarray(population, dtype=np.float64)
    if population.ndim != 3 or population.shape[2] != 2:
        raise ValueError(
            "population must have shape (population_size, num_turbines, 2)."
        )

    wind_speeds = np.asarray(wind_speeds, dtype=np.float64).reshape(-1)
    wind_directions = np.asarray(wind_directions, dtype=np.float64).reshape(-1)
    if wind_speeds.size == 0:
        raise ValueError("At least one wind scenario is required.")
    if wind_directions.size != wind_speeds.size:
        raise ValueError("wind_speeds and wind_directions must have equal lengths.")

    turbulence_intensities = np.asarray(turbulence_intensities, dtype=np.float64)
    if turbulence_intensities.ndim == 0:
        turbulence_intensities = np.full(
            wind_speeds.size,
            float(turbulence_intensities),
            dtype=np.float64,
        )
    else:
        turbulence_intensities = turbulence_intensities.reshape(-1)
    if turbulence_intensities.size != wind_speeds.size:
        raise ValueError(
            "turbulence_intensities must be scalar or contain one value per scenario."
        )

    pop_size, n_turbines, _ = population.shape
    n_scenarios = wind_speeds.size

    config_path = Path(config_path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"FLORIS configuration not found: {config_path}")

    powers_kw = np.empty((pop_size, n_scenarios), dtype=np.float64)
    waked_velocities = np.empty((pop_size, n_scenarios, n_turbines), dtype=np.float64)

    tasks = [
        (
            i,
            population[i],
            wind_speeds,
            wind_directions,
            turbulence_intensities,
        )
        for i in range(pop_size)
    ]

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_initialize_floris_worker,
        initargs=(str(config_path), turbine_type),
    ) as executor:
        iterator = executor.map(
            _evaluate_layout_worker,
            tasks,
            chunksize=chunksize,
        )
        if show_progress:
            iterator = tqdm(
                iterator,
                total=pop_size,
                desc="FLORIS layouts",
                position=1,
                leave=False,
            )

        for layout_index, layout_powers, layout_velocities in iterator:
            powers_kw[layout_index] = layout_powers
            waked_velocities[layout_index] = layout_velocities

    return {
        "powers_kw": powers_kw,
        "waked_velocities": waked_velocities,
    }
