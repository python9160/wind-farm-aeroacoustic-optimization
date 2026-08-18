"""
core_checker.py

Quick system check for choosing FLORIS parallelization settings.

Run:
    python core_checker.py
"""

import os
import platform
import subprocess


def get_cpu_name():
    """Return a human-readable CPU name when available."""

    system = platform.system()

    try:
        if system == "Windows":
            result = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "(Get-CimInstance Win32_Processor).Name",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()

        elif system == "Linux":
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":", 1)[1].strip()

        elif system == "Darwin":
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()

    except Exception:
        pass

    return platform.processor() or "Unknown"


def get_physical_cores():
    """Try to determine the number of physical CPU cores."""

    system = platform.system()

    try:
        if system == "Windows":
            result = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    (
                        "(Get-CimInstance Win32_Processor | "
                        "Measure-Object -Property NumberOfCores -Sum).Sum"
                    ),
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            return int(result.stdout.strip())

        elif system == "Linux":
            result = subprocess.run(
                ["lscpu", "-p=CORE,SOCKET"],
                capture_output=True,
                text=True,
                check=True,
            )

            cores = {
                line
                for line in result.stdout.splitlines()
                if line and not line.startswith("#")
            }

            return len(cores)

        elif system == "Darwin":
            result = subprocess.run(
                ["sysctl", "-n", "hw.physicalcpu"],
                capture_output=True,
                text=True,
                check=True,
            )

            return int(result.stdout.strip())

    except Exception:
        return None

    return None


def recommend_settings(physical_cores, logical_cpus):
    """
    Produce conservative starting values for the FLORIS process pool.

    These are recommendations only; optimal settings depend on CPU
    architecture, memory, FLORIS workload, and other running processes.
    """

    if physical_cores is not None:
        # Keep roughly 1-2 physical cores free.
        if physical_cores <= 2:
            workers = 1
        elif physical_cores <= 4:
            workers = physical_cores - 1
        else:
            workers = physical_cores - 2
    else:
        # Fall back to logical CPU count.
        workers = max(
            1,
            int(logical_cpus * 0.6),
        )

    # Your current value is a sensible general starting point.
    chunksize = 4

    return workers, chunksize


def main():

    cpu_name = get_cpu_name()

    logical_cpus = os.cpu_count() or 1
    physical_cores = get_physical_cores()

    workers, chunksize = recommend_settings(
        physical_cores,
        logical_cpus,
    )

    print()
    print("=" * 64)
    print(" FLORIS PARALLELIZATION SYSTEM CHECK")
    print("=" * 64)

    print()
    print("System")
    print("-" * 64)
    print(f"Operating system : {platform.system()} {platform.release()}")
    print(f"Architecture     : {platform.machine()}")
    print(f"Python version   : {platform.python_version()}")

    print()
    print("Processor")
    print("-" * 64)
    print(f"CPU              : {cpu_name}")

    if physical_cores is not None:
        print(f"Physical cores   : {physical_cores}")
    else:
        print("Physical cores   : Could not determine")

    print(f"Logical CPUs     : {logical_cpus}")

    if physical_cores:
        print(f"Threads / core   : " f"{logical_cpus / physical_cores:.2f}")

    print()
    print("Recommended FLORIS Settings")
    print("-" * 64)

    print(f"MAX_WORKERS      = {workers}")

    print(f"FLORIS_CHUNKSIZE = {chunksize}")

    print()
    print("Copy into GA_with_OpenFAST_validation.ipynb:")
    print()

    print(f"MAX_WORKERS = {workers}")

    print(f"FLORIS_CHUNKSIZE = {chunksize}")

    print()
    print("Notes")
    print("-" * 64)

    print("* MAX_WORKERS is a conservative starting recommendation.")

    print(
        "* Some CPU capacity is intentionally left available for the "
        "OS and main Python process."
    )

    print(
        "* Hybrid CPUs (for example, Intel P-core/E-core designs) may "
        "require manual tuning."
    )

    print(
        "* If the system becomes unresponsive or memory usage is high, "
        "reduce MAX_WORKERS."
    )

    print(
        "* Benchmarking nearby worker counts is the best way to determine "
        "the optimal value for a particular machine."
    )

    print()
    print("=" * 64)
    print()


if __name__ == "__main__":
    main()
