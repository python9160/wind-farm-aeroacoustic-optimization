import io
import re
import pandas as pd


def lout(filepath: str) -> pd.DataFrame:
    """Reads a tab-delimited file, combines the headers and units using pure Python,

    and returns a clean, fully numeric DataFrame.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 1. Extract the raw text headers and units (accounting for 0-indexed skipping)
    #    Line 6 (7th line) = Header Names, Line 7 (8th line) = Units
    header_names = lines[6].strip().split("\t")
    header_units = lines[7].strip().split("\t")

    # 2. Combine them into a single tab-separated header string
    #    Handles mismatches gracefully by pairing elements up
    combined_headers = []
    for name, unit in zip(header_names, header_units):
        combined_headers.append(f"{name.strip()} {unit.strip()}")

    new_header_line = "\t".join(combined_headers) + "\n"

    # 3. Gather all the actual numerical data lines (Line 8 onwards)
    data_lines = lines[8:]

    # 4. Reconstruct the clean table purely out of text strings
    clean_text_data = "".join([new_header_line] + data_lines)

    # 5. Pass the clean stream directly into Pandas
    #    Pandas will now automatically infer proper numeric dtypes (float64/int64)
    df = pd.read_csv(io.StringIO(clean_text_data), sep="\t")

    if "Time (s)" in df.columns:
        df = df.set_index("Time (s)")

    return df


import os
from pathlib import Path


def aggregate_aweighted_oaspl(
    folder_path: str | Path,
    glob="*AD.AA1.out",
) -> pd.DataFrame:
    """
    Find all matching OpenFAST aeroacoustic output files and energetically
    combine their already A-weighted observer OASPL values.

    Expected acoustic columns look like:

        Obs1 OASPL
        Obs2 OASPL
        Obs3 OASPL
        ...

    Each file is assumed to represent one acoustic source/turbine.
    The observer levels are summed energetically across all files:

        L_total = 10 * log10(sum(10 ** (L_i / 10)))

    Parameters
    ----------
    folder_path : str or Path
        Folder containing the OpenFAST AA output files.

    glob : str
        File pattern used to locate output files.

    Returns
    -------
    pd.DataFrame
        Metadata columns from the first file plus the energetically
        aggregated observer OASPL columns.
    """
    folder_path = Path(folder_path)

    target_files = sorted(folder_path.glob(glob))

    if not target_files:
        raise ValueError(f"No files matching {glob!r} found in {folder_path}")

    # ---------------------------------------------------------
    # Load the first file
    # ---------------------------------------------------------
    first_df = lout(target_files[0])

    base_df = first_df.copy()

    # Observer columns such as:
    #   Obs1 OASPL
    #   Obs2 OASPL
    oaspl_cols = [
        column for column in base_df.columns if str(column).strip().endswith("OASPL")
    ]

    if not oaspl_cols:
        raise ValueError(
            f"No OASPL columns found in {target_files[0]}. "
            f"Columns were: {base_df.columns.tolist()}"
        )

    # ---------------------------------------------------------
    # Initialize energetic sum
    # ---------------------------------------------------------
    linear_sum = np.power(
        10.0,
        base_df[oaspl_cols].to_numpy(dtype=np.float64) / 10.0,
    )

    # ---------------------------------------------------------
    # Add remaining turbine/source files energetically
    # ---------------------------------------------------------
    for filepath in target_files[1:]:
        current_df = lout(filepath)

        missing_columns = [
            column for column in oaspl_cols if column not in current_df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"{filepath} is missing expected OASPL columns: " f"{missing_columns}"
            )

        if len(current_df) != len(base_df):
            raise ValueError(
                f"Row count mismatch for {filepath}. "
                f"Expected {len(base_df)}, got {len(current_df)}."
            )

        linear_sum += np.power(
            10.0,
            current_df[oaspl_cols].to_numpy(dtype=np.float64) / 10.0,
        )

    # ---------------------------------------------------------
    # Convert total acoustic energy back to dBA
    # ---------------------------------------------------------
    combined_oaspl = 10.0 * np.log10(
        np.maximum(
            linear_sum,
            np.finfo(np.float64).tiny,
        )
    )

    combined_df = pd.DataFrame(
        combined_oaspl,
        index=base_df.index,
        columns=oaspl_cols,
    )

    # ---------------------------------------------------------
    # Keep non-acoustic metadata from the first file
    # ---------------------------------------------------------
    metadata_cols = [column for column in base_df.columns if column not in oaspl_cols]

    final_df = pd.concat(
        [
            base_df[metadata_cols],
            combined_df,
        ],
        axis=1,
    )

    return final_df


def aggregate_oaspl(folder_path: str | Path, glob="*AD.AA2.out") -> pd.DataFrame:
    """
    Finds all '*AD.AA2.out' files in the folder, loads them with lout(),
    and computes the total combined OASPL across all files.
    """
    # 1. Cast to Path object (handles both str and Path instances seamlessly)
    folder_path = Path(folder_path)

    # Use pathlib's glob method
    target_files = list(folder_path.glob(glob))

    if not target_files:
        raise ValueError(f"No files matching '{glob}' found in {folder_path}")

    # 2. Initialize using the first file
    first_df = lout(target_files[0])
    base_df = first_df.copy()
    spl_cols = base_df.filter(regex="SPL$").columns

    # Initialize our running linear sum
    linear_sum = 10 ** (base_df[spl_cols] / 10)

    # 3. Loop through the remaining files and accumulate in the linear domain
    for filepath in target_files[1:]:
        current_df = lout(filepath)
        linear_sum += 10 ** (current_df[spl_cols] / 10)

    # 4. Group columns by observer (e.g., 'Obs1_...' -> 'Obs1')
    observer_groups = linear_sum.columns.str.split("_").str[0]

    # 5. Sum the combined linear energies by group and convert back to dB (OASPL)
    oaspl_df = 10 * np.log10(linear_sum.T.groupby(observer_groups).sum().T)

    # 6. Rename columns to denote they are OASPL
    oaspl_df.columns = [f"{col} OASPL" for col in oaspl_df.columns]

    # 7. Merge the final calculated OASPL back with the original metadata
    metadata_cols = base_df.columns.difference(spl_cols)
    final_df = base_df[metadata_cols].join(oaspl_df)

    return final_df


import numpy as np
import pandas as pd
from pathlib import Path

# Pre-computed precise IEC 61672-1 A-weighting adjustments for your 34 specific bands
A_WEIGHT_MAP = {
    10.0: -70.4,
    12.5: -63.4,
    16.0: -56.7,
    20.0: -50.5,
    25.0: -44.7,
    31.5: -39.4,
    40.0: -34.6,
    50.0: -30.2,
    63.0: -26.2,
    80.0: -22.5,
    100.0: -19.1,
    125.0: -16.1,
    160.0: -13.4,
    200.0: -10.9,
    250.0: -8.6,
    315.0: -6.6,
    400.0: -4.8,
    500.0: -3.2,
    630.0: -1.9,
    800.0: -0.8,
    1000.0: 0.0,
    1250.0: 0.6,
    1600.0: 1.0,
    2000.0: 1.2,
    2500.0: 1.3,
    3150.0: 1.2,
    4000.0: 1.0,
    5000.0: 0.5,
    6300.0: -0.1,
    8000.0: -1.1,
    10000.0: -2.5,
    12500.0: -4.3,
    16000.0: -6.6,
    20000.0: -9.3,
}


def aggregate_dba_optimized(
    folder_path: str | Path, only_aggregate: bool = True
) -> pd.DataFrame | dict[str, pd.DataFrame]:
    """
    Finds all '*AD.AA2.out' files, extracts frequency weights efficiently using a
    vectorized lookup map, and computes A-weighted sound pressure levels (dBA).

    Parameters
    ----------
    folder_path : str | Path
        Path to the folder containing OpenFAST output files.
    only_aggregate : bool, default True
        If True, returns a single DataFrame representing combined noise across all turbines.
        If False, returns a dict with:
            - 'aggregate': Combined dBA DataFrame across all turbines.
            - 'individuals': Dict mapping turbine IDs (e.g., 'T1', 'T2') to their
                             individual dBA DataFrames.

    Returns
    -------
    pd.DataFrame | dict
        Combined DataFrame or dictionary containing aggregate + individual DataFrames.
    """
    folder_path = Path(folder_path)
    target_files = list(folder_path.glob("*AD.AA2.out"))

    if not target_files:
        raise ValueError(f"No files matching '*AD.AA2.out' found in {folder_path}")

    # Helper function to compute dBA for a single raw SPL DataFrame
    def process_file_dba(
        df: pd.DataFrame, spl_cols: pd.Index, a_adjust: pd.Series
    ) -> pd.DataFrame:
        weighted_spl = df[spl_cols] + a_adjust
        linear_val = 10 ** (weighted_spl / 10)

        # Group by observer (e.g. 'Obs1' from 'Obs1_Freq10 SPL')
        observer_groups = linear_val.columns.str.split("_").str[0]
        dba = 10 * np.log10(linear_val.T.groupby(observer_groups).sum().T)
        dba.columns = [f"{col} dBA" for col in dba.columns]
        return dba

    # 1. Initialize using the first file
    base_df = lout(target_files[0]).copy()
    spl_cols = base_df.filter(regex="SPL$").columns

    # 2. Vectorized Frequency Extraction & Mapping
    extracted_freqs = (
        spl_cols.str.split("_Freq").str[1].str.split(" ").str[0].astype(float)
    )
    a_adjust_series = pd.Series(extracted_freqs).map(A_WEIGHT_MAP).fillna(0.0)
    a_adjust_series.index = spl_cols  # Align index back with column names

    # Metadata columns (Time, etc.) to keep alongside observer dBA
    metadata_cols = base_df.columns.difference(spl_cols)
    meta_df = base_df[metadata_cols]

    # Store individual turbine linear values and calculated dBA
    individual_linear = {}
    individual_dba_dfs = {}

    for filepath in target_files:
        # Extract turbine identifier (e.g. 'T1', 'T2' from 'test.T1.AD.AA2.out')
        match = re.search(r"\.T(\d+)\.AD\.AA2\.out$", filepath.name)
        turbine_id = f"WT{int(match.group(1)) - 1}" if match else filepath.stem

        current_df = lout(filepath)

        # Process individual dBA for this turbine
        turbine_dba = process_file_dba(current_df, spl_cols, a_adjust_series)
        individual_dba_dfs[turbine_id] = meta_df.join(turbine_dba)

        # Save linear values for aggregate sum: 10^(dBA / 10)
        individual_linear[turbine_id] = 10 ** (turbine_dba / 10)

    # 3. Sum combined linear energies across all turbines
    aggregate_linear_sum = sum(individual_linear.values())
    aggregate_dba = 10 * np.log10(aggregate_linear_sum)
    aggregate_df = meta_df.join(aggregate_dba)

    # 4. Return based on flag
    if only_aggregate:
        return aggregate_df

    return {"aggregate": aggregate_df, "individuals": individual_dba_dfs}


import json

from typing import Any, Dict, Optional, overload


# Signature 1: Passing a single Path or str
@overload
def save_metadata(path: str | Path, **kwargs: Any) -> str: ...


# Signature 2: Passing separate directory and trial name strings
@overload
def save_metadata(save_dir: str, trial_name: str, **kwargs: Any) -> str: ...


# Runtime Implementation
def save_metadata(
    save_dir_or_path: str | Path, trial_name: Optional[str] = None, **kwargs: Any
) -> str:
    """Saves trial metadata to target path or save_dir/trial_name/metadata.json."""
    if isinstance(save_dir_or_path, Path):
        target_path = save_dir_or_path
    elif isinstance(save_dir_or_path, str) and isinstance(trial_name, str):
        target_path = Path(save_dir_or_path) / trial_name
    elif isinstance(save_dir_or_path, str) and trial_name is None:
        target_path = Path(save_dir_or_path)
    else:
        raise TypeError("Invalid argument combination for save_metadata.")

    # Ensure directory exists and target the json file
    target_path.mkdir(parents=True, exist_ok=True)
    filepath = target_path / "metadata.json"

    # Serialize NumPy arrays to nested lists
    serialized_data = {
        k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in kwargs.items()
    }

    with open(filepath, "w") as f:
        json.dump(serialized_data, f, indent=4)

    return str(filepath)


# ==============================================================================
# load_metadata
# ==============================================================================


# Signature 1: Passing a single Path or str
@overload
def load_metadata(path: str | Path) -> Dict[str, Any]: ...


# Signature 2: Passing separate directory and trial name strings
@overload
def load_metadata(save_dir: str, trial_name: str) -> Dict[str, Any]: ...


# Runtime Implementation
def load_metadata(
    save_dir_or_path: str | Path, trial_name: Optional[str] = None
) -> Dict[str, Any]:
    """Loads trial metadata from target path or save_dir/trial_name/metadata.json,

    automatically converting lists back into NumPy arrays.
    """
    if isinstance(save_dir_or_path, Path):
        target_path = save_dir_or_path
    elif isinstance(save_dir_or_path, str) and isinstance(trial_name, str):
        target_path = Path(save_dir_or_path) / trial_name
    elif isinstance(save_dir_or_path, str) and trial_name is None:
        target_path = Path(save_dir_or_path)
    else:
        raise TypeError("Invalid argument combination for load_metadata.")

    filepath = target_path / "metadata.json"

    if not filepath.exists():
        raise FileNotFoundError(f"Metadata file not found at: {filepath}")

    with open(filepath, "r") as f:
        data = json.load(f)

    # Convert any list values back into NumPy arrays
    parsed_data = {}
    for key, value in data.items():
        if isinstance(value, list):
            parsed_data[key] = np.array(value)
        else:
            parsed_data[key] = value

    return parsed_data


import sys


def is_notebook() -> bool:
    """
    Returns True if running in a Jupyter notebook/lab environment,
    and False if running in a standard terminal script.
    """
    if "IPython" not in sys.modules:
        return False

    from IPython import get_ipython

    shell = get_ipython()

    if shell is None:
        return False

    shell_name = shell.__class__.__name__
    return shell_name in ("ZMQInteractiveShell", "Shell")


# Pre-evaluate the environment once during module import
_IN_NOTEBOOK = is_notebook()


def jprint(*args, **kwargs):
    """
    Prints content normally if executed in a Jupyter notebook.
    Silences all output completely when run as a standalone terminal script.
    """
    return

    if _IN_NOTEBOOK:
        print(*args, **kwargs)


import numpy as np


def generar_puntos_circulo(
    n_puntos: int = 16, radio: float = 1.0, z_constante: float = 0.0
) -> np.ndarray:
    """
    Genera N puntos equidistantes en el círculo con un radio especifico (plano X-Y)
    y les asigna una coordenada Z constante.

    Retorna un array de NumPy con forma (n_puntos, 3).
    """
    # 1. Generar los ángulos distribuidos uniformemente de 0 a 2pi.
    # endpoint=False evita que el último punto coincida con el primero (0 y 360°).
    angulos = np.linspace(0, 2 * np.pi, n_puntos, endpoint=False)

    # 2. Calcular las coordenadas en el plano X-Y
    x = np.cos(angulos) * radio
    y = np.sin(angulos) * radio

    # 3. Crear un vector de Z constantes con el mismo tamaño
    z = np.full_like(x, z_constante)

    # 4. Apilar las columnas para formar una matriz de vectores [X, Y, Z]
    puntos_3d = np.column_stack((x, y, z))

    return puntos_3d


from typing import List, Tuple, Union, overload
import numpy as np
import numpy.typing as npt


@overload
def rotate_layout_for_openfast(
    points: npt.ArrayLike, *, angle: float = 270.0
) -> np.ndarray: ...


@overload
def rotate_layout_for_openfast(
    *point_groups: npt.ArrayLike, angle: float = 270.0
) -> List[np.ndarray]: ...


def rotate_layout_for_openfast(
    *point_groups: npt.ArrayLike, angle: float = 270.0
) -> Union[np.ndarray, List[np.ndarray]]:
    """
    Rotates one or more sets of layout coordinates to simulate varying
    FLORIS wind directions while keeping OpenFAST fixed at PropagationDir = 0.
    """
    if not point_groups:
        raise ValueError("At least one set of points must be provided.")

    # Convert all inputs to numpy arrays explicitly for safe indexing and length calculations
    np_groups: Tuple[np.ndarray, ...] = tuple(np.asarray(g) for g in point_groups)

    # 1. Combine all point arrays into a single contiguous array
    combined_points: np.ndarray = np.r_[np_groups]

    # 2. Calculate rotation matrix
    delta_theta_deg: float = angle - 270.0
    theta_rad: float = np.radians(delta_theta_deg)
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    R: np.ndarray = np.array([[c, s], [-s, c]])

    # 3. Center calculation based on ALL combined coordinates
    center: np.ndarray = (
        combined_points.max(axis=0) + combined_points.min(axis=0)
    ) / 2.0

    # 4. Rotate the entire combined set
    rotated_combined: np.ndarray = np.dot(combined_points - center, R) + center

    # 5. Determine split indices using cumulative lengths
    lengths: List[int] = [len(group) for group in np_groups[:-1]]
    split_indices: np.ndarray = np.cumsum(lengths)

    # 6. Split back into original group shapes
    rotated_groups: List[np.ndarray] = np.split(rotated_combined, split_indices)

    # Return a single array if only one input was passed, otherwise a list
    return rotated_groups[0] if len(np_groups) == 1 else rotated_groups
