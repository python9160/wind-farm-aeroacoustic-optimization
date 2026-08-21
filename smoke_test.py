"""
smoke_test.py

Basic installation and project-structure verification for the
OpenFAST wind farm aeroacoustic optimization project.

Run from the project root:

    python smoke_test.py

This script does not run OpenFAST, FAST.Farm, FLORIS simulations,
or the Genetic Algorithm. It only verifies that the main dependencies,
files, executables, and project modules can be found/imported.
"""

from pathlib import Path
import importlib
import sys

# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

REQUIRED_EXECUTABLES = [
    "OpenFAST.exe",
    "FAST.Farm.exe",
]

OPTIONAL_EXECUTABLES = [
    "TurbSim.exe",
]

REQUIRED_DIRECTORIES = [
    "5MW",
    "5MW_farm",
    "wrapper",
    "dba_model_outputs",
]

REQUIRED_FILES = [
    "training_data.csv",
    "GA_with_OpenFAST_validation.ipynb",
    "wrapper/__init__.py",
    "wrapper/gch.yaml",
]

OPTIONAL_FILES = [
    "noise_lookup_table.json",
    "final_train.ipynb",
    "core_checker.py",
]

REQUIRED_PACKAGES = [
    "numpy",
    "pandas",
    "matplotlib",
    "scipy",
    "sklearn",
    "tqdm",
    "joblib",
    "floris",
    "xgboost",
]

# openfast_toolbox may be installed differently depending on
# how the official repository was configured.
OPENFAST_TOOLBOX_MODULE = "openfast_toolbox"

WRAPPER_MODULES = [
    "wrapper",
    "wrapper.openfast_base",
    "wrapper.utils",
    "wrapper.main",
    "wrapper.farm",
    "wrapper.aa_surrogate_ml_vectorized",
    "wrapper.parallel_floris",
    "wrapper.parallel_floris_class",
]


# ============================================================
# ANSI terminal formatting
# ============================================================

# Disable ANSI colors automatically if stdout is being redirected
# to a file or another non-interactive destination.
USE_COLOR = sys.stdout.isatty()


def ansi(code):
    """Return ANSI escape code only when terminal coloring is enabled."""
    return code if USE_COLOR else ""


RESET = ansi("\033[0m")
BOLD = ansi("\033[1m")
DIM = ansi("\033[2m")

RED = ansi("\033[31m")
GREEN = ansi("\033[32m")
YELLOW = ansi("\033[33m")
CYAN = ansi("\033[36m")

PASS = f"{GREEN}[PASS]{RESET}"
FAIL = f"{RED}[FAIL]{RESET}"
WARN = f"{YELLOW}[WARN]{RESET}"


# ============================================================
# Formatting helpers
# ============================================================


def section(title):
    """Print a formatted section heading."""
    print()
    print(f"{CYAN}{'=' * 68}{RESET}")
    print(f"{BOLD}{CYAN}{title}{RESET}")
    print(f"{CYAN}{'=' * 68}{RESET}")


def check_path(path, required=True):
    """Check whether a file or directory exists."""

    relative = path.relative_to(PROJECT_ROOT)

    if path.exists():
        print(f"{PASS} {relative}")
        return True

    status = FAIL if required else WARN

    print(f"{status} {relative} " f"{DIM}-- not found{RESET}")

    return False


def check_import(module_name, required=True):
    """Attempt to import a Python module."""

    try:
        module = importlib.import_module(module_name)

        version = getattr(
            module,
            "__version__",
            None,
        )

        if version:
            print(f"{PASS} {module_name:<35} " f"{DIM}version {version}{RESET}")
        else:
            print(f"{PASS} {module_name}")

        return True

    except Exception as exc:
        status = FAIL if required else WARN

        print(f"{status} {module_name:<35} " f"{DIM}{type(exc).__name__}: {exc}{RESET}")

        return False


# ============================================================
# Main smoke test
# ============================================================


def main():

    print()
    print(f"{CYAN}{'=' * 68}{RESET}")
    print(f"{BOLD}{CYAN}" " OpenFAST Project Smoke Test" f"{RESET}")
    print(f"{CYAN}{'=' * 68}{RESET}")

    print()
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Python       : {sys.version.split()[0]}")
    print(f"Executable   : {sys.executable}")

    failures = 0
    warnings = 0

    # --------------------------------------------------------
    # Required directories
    # --------------------------------------------------------

    section("Required Directories")

    for relative_path in REQUIRED_DIRECTORIES:

        path = PROJECT_ROOT / relative_path

        if not check_path(path, required=True):
            failures += 1

    # --------------------------------------------------------
    # Required project files
    # --------------------------------------------------------

    section("Required Project Files")

    for relative_path in REQUIRED_FILES:

        path = PROJECT_ROOT / relative_path

        if not check_path(path, required=True):
            failures += 1

    # --------------------------------------------------------
    # Optional project files
    # --------------------------------------------------------

    section("Optional Project Files")

    for relative_path in OPTIONAL_FILES:

        path = PROJECT_ROOT / relative_path

        if not check_path(path, required=False):
            warnings += 1

    # --------------------------------------------------------
    # OpenFAST executables
    # --------------------------------------------------------

    section("OpenFAST Executables")

    for executable in REQUIRED_EXECUTABLES:

        path = PROJECT_ROOT / executable

        if not check_path(path, required=True):
            failures += 1

    for executable in OPTIONAL_EXECUTABLES:

        path = PROJECT_ROOT / executable

        if not check_path(path, required=False):
            warnings += 1

    # --------------------------------------------------------
    # Python dependencies
    # --------------------------------------------------------

    section("Python Dependencies")

    for package in REQUIRED_PACKAGES:

        if not check_import(package, required=True):
            failures += 1

    # --------------------------------------------------------
    # OpenFAST Toolbox
    # --------------------------------------------------------

    section("OpenFAST Toolbox")

    if not check_import(
        OPENFAST_TOOLBOX_MODULE,
        required=True,
    ):
        failures += 1

    # --------------------------------------------------------
    # Wrapper package
    # --------------------------------------------------------

    section("Project Wrapper")

    for module in WRAPPER_MODULES:

        if not check_import(module, required=True):
            failures += 1

    # --------------------------------------------------------
    # FLORIS configuration
    # --------------------------------------------------------

    section("FLORIS Configuration")

    gch_path = PROJECT_ROOT / "wrapper" / "gch.yaml"

    if not check_path(
        gch_path,
        required=True,
    ):
        failures += 1

    # --------------------------------------------------------
    # ML surrogate model
    # --------------------------------------------------------

    section("ML Surrogate Model")

    model_dir = PROJECT_ROOT / "dba_model_outputs"

    if not model_dir.exists():

        print(f"{FAIL} dba_model_outputs/ " f"{DIM}-- directory not found{RESET}")

        failures += 1

    else:

        print(f"{PASS} dba_model_outputs/")

        model_files = sorted(model_dir.glob("*.joblib"))

        if model_files:

            for model_file in model_files:

                relative = model_file.relative_to(PROJECT_ROOT)

                print(f"{PASS} Model found: {relative}")

        else:

            print(f"{FAIL} No .joblib model found in " f"dba_model_outputs/")

            failures += 1

    # --------------------------------------------------------
    # Training data
    # --------------------------------------------------------

    section("Training Data")

    training_data = PROJECT_ROOT / "training_data.csv"

    if training_data.exists():

        try:
            import pandas as pd

            # Only read a few rows. This is a smoke test,
            # not a full validation of the training dataset.
            df = pd.read_csv(
                training_data,
                nrows=5,
            )

            print(f"{PASS} training_data.csv readable")

            print(f"       {DIM}Columns: " f"{list(df.columns)}{RESET}")

            expected_columns = {
                "v",
                "r",
                "cos_theta",
                "sin_theta",
                "dBA",
            }

            missing = expected_columns - set(df.columns)

            if missing:

                print(f"{FAIL} Missing expected columns: " f"{sorted(missing)}")

                failures += 1

            else:

                print(f"{PASS} Required ML columns present")

        except Exception as exc:

            print(
                f"{FAIL} Could not read training_data.csv: "
                f"{DIM}{type(exc).__name__}: "
                f"{exc}{RESET}"
            )

            failures += 1

    else:

        # This should already have been caught in the required
        # project files section, so avoid incrementing failures
        # a second time.
        print(f"{FAIL} training_data.csv not available " f"for validation")

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    section("Summary")

    failure_color = RED if failures else GREEN

    warning_color = YELLOW if warnings else GREEN

    print("Failures : " f"{failure_color}{failures}{RESET}")

    print("Warnings : " f"{warning_color}{warnings}{RESET}")

    print()

    # --------------------------------------------------------
    # Successful result
    # --------------------------------------------------------

    if failures == 0:

        print(f"{BOLD}{GREEN}" "Smoke test PASSED." f"{RESET}")

        print()

        print(
            "The required project files, Python dependencies, "
            "OpenFAST executables, and wrapper modules were found."
        )

        if warnings:

            print()

            print(
                f"{YELLOW}"
                "Optional components are missing; "
                "see warnings above."
                f"{RESET}"
            )

        print()

        print(
            f"{GREEN}"
            "You should be ready to configure and run "
            "GA_with_OpenFAST_validation.ipynb."
            f"{RESET}"
        )

        print()

        return 0

    # --------------------------------------------------------
    # Failed result
    # --------------------------------------------------------

    print(f"{BOLD}{RED}" "Smoke test FAILED." f"{RESET}")

    print()

    print(
        f"{RED}"
        "Resolve the failed checks above before running "
        "the optimization workflow."
        f"{RESET}"
    )

    print()

    return 1


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    raise SystemExit(main())
