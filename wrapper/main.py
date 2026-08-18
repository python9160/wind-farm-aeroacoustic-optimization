import os
import sys
import re
import shutil
import subprocess
import fnmatch

from wrapper.openfast_base import OpenFASTFile

from .farm import Farm


def run_openfast_old(fst_path, save_dir=".", custom_name=None, progress_callback=None):
    """
    Runs an OpenFAST simulation using a live stream reader to capture stdout.
    Optionally reports completion tracking to an ipywidget callback.
    """
    openfast_exe = os.path.abspath("OpenFAST.exe")
    if not os.path.exists(openfast_exe):
        raise FileNotFoundError(f"Could not find local executable at: {openfast_exe}")

    target_fst_path = os.path.abspath(fst_path)
    if not os.path.exists(target_fst_path):
        raise FileNotFoundError(
            f"The OpenFAST file '{target_fst_path}' does not exist."
        )

    target_dir = os.path.dirname(target_fst_path)
    fst_filename = os.path.basename(target_fst_path)
    fst_base_name = os.path.splitext(fst_filename)[0]

    os.makedirs(save_dir, exist_ok=True)

    # Regular expression to extract progress tracking variables from terminal text
    # OpenFAST logs typically look like: " Time:     10 of    60 seconds."
    progress_regex = re.compile(r"Time:\s+([\d.]+)\s+of\s+([\d.]+)\s+seconds")

    # Launch OpenFAST using Popen with stdout routing turned on
    process = subprocess.Popen(
        [openfast_exe, fst_filename],
        cwd=target_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # Merge error messages into the standard stream
        text=True,
        bufsize=1,  # Line-buffered reading for real-time streaming
    )

    # Continuously parse the live command window outputs
    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break  # Exit loop when the executable terminates cleanly

        if line:
            # Uncomment the next line if you still want to mirror it to the terminal log
            # print(line.strip())

            # Look for time step progress flags
            match = progress_regex.search(line)
            if match and progress_callback:
                current_time = float(match.group(1))
                total_time = float(match.group(2))

                # Send the clean numeric float back to our widget listener
                progress_callback(current_time, total_time)

    if process.returncode != 0:
        raise RuntimeError(f"OpenFAST exited with error code {process.returncode}")

    # File moving routine remains identical
    moved_files = []
    for file in os.listdir(target_dir):
        if file.startswith(fst_base_name) and file.endswith(".out"):
            source_file_path = os.path.join(target_dir, file)
            final_name = f"{custom_name}.out" if custom_name else file
            destination_file_path = os.path.join(save_dir, final_name)

            shutil.move(source_file_path, destination_file_path)
            moved_files.append(destination_file_path)

    return moved_files


def run_openfast(fst_path, save_dir=".", trial_name=None, progress_callback=None):
    """
    Runs an OpenFAST simulation, automatically identifying and capturing ALL
    newly generated files, moving them into a dedicated trial subfolder.
    """
    openfast_exe = os.path.abspath("OpenFAST.exe")
    if not os.path.exists(openfast_exe):
        raise FileNotFoundError(f"Could not find local executable at: {openfast_exe}")

    target_fst_path = os.path.abspath(fst_path)
    if not os.path.exists(target_fst_path):
        raise FileNotFoundError(
            f"The OpenFAST file '{target_fst_path}' does not exist."
        )

    target_dir = os.path.dirname(target_fst_path)
    fst_filename = os.path.basename(target_fst_path)

    # Define and create a distinct subfolder inside your save directory for this specific trial
    trial_folder_name = trial_name if trial_name else os.path.splitext(fst_filename)[0]
    final_output_dir = os.path.join(save_dir, trial_folder_name)
    os.makedirs(final_output_dir, exist_ok=True)

    # --------------------------------------------------------------------------
    # SNAPSHOT 1: Record existing files before execution
    # --------------------------------------------------------------------------
    pre_existing_files = set(os.listdir(target_dir))

    # Progress tracking regex setup
    progress_regex = re.compile(r"Time:\s+([\d.]+)\s+of\s+([\d.]+)\s+seconds")

    # Launch OpenFAST
    process = subprocess.Popen(
        [openfast_exe, fst_filename],
        cwd=target_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    # Live console stream reader
    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break

        if line:
            match = progress_regex.search(line)
            if match and progress_callback:
                current_time = float(match.group(1))
                total_time = float(match.group(2))
                progress_callback(current_time, total_time)

    if process.returncode != 0:
        raise RuntimeError(f"OpenFAST exited with error code {process.returncode}")

    # --------------------------------------------------------------------------
    # SNAPSHOT 2: Isolate and move newly generated artifacts
    # --------------------------------------------------------------------------
    post_execution_files = set(os.listdir(target_dir))
    new_files = post_execution_files - pre_existing_files

    moved_files = []
    for file in new_files:
        source_file_path = os.path.join(target_dir, file)

        # Guard clause: Ensure we are only moving files, not nested baseline directories
        if os.path.isfile(source_file_path):
            destination_file_path = os.path.join(final_output_dir, file)

            shutil.move(source_file_path, destination_file_path)
            moved_files.append(destination_file_path)

    return moved_files


import fnmatch
import os
import re
import shutil
import subprocess


def run_fastfarm(
    fstf: str | Farm | OpenFASTFile,
    save_dir=".",
    trial_name: str = None,
    progress_callback=None,
    files_to_save=None,
):
    """Runs a FAST.Farm simulation, automatically capturing only specified files

    and moving them into a dedicated trial subfolder.
    """
    # Safe default for regex patterns matching *.AA2.out and T<number>.out
    if files_to_save is None:
        files_to_save = [
            r".*[.]AA1[.]out$",  # Matches any filename ending with .AA2.out
            r".*[.]T\d+[.]out$",  # Matches .T<number>.out pattern
        ]

    fastfarm_exe = os.path.abspath("FAST.Farm.exe")
    if not os.path.exists(fastfarm_exe):
        raise FileNotFoundError(f"Could not find local executable at: {fastfarm_exe}")

    fstf_path = (
        fstf.filepath
        if isinstance(fstf, Farm) or isinstance(fstf, OpenFASTFile)
        else fstf
    )
    target_fstf_path = os.path.abspath(fstf_path)
    if not os.path.exists(target_fstf_path):
        raise FileNotFoundError(
            f"The FAST.Farm file '{target_fstf_path}' does not exist."
        )

    target_dir = os.path.dirname(target_fstf_path)
    fstf_filename = os.path.basename(target_fstf_path)

    # Define and create a distinct subfolder inside your save directory
    trial_folder_name = trial_name if trial_name else os.path.splitext(fstf_filename)[0]
    final_output_dir = os.path.join(save_dir, trial_folder_name)
    os.makedirs(final_output_dir, exist_ok=True)

    # Compile the files_to_save regex patterns
    save_patterns = [re.compile(pat) for pat in files_to_save]

    # --------------------------------------------------------------------------
    # SNAPSHOT 1: Record existing files before execution
    # --------------------------------------------------------------------------
    pre_existing_files = set(os.listdir(target_dir))

    # Progress tracking regex setup (Matches: "Time: 10 of 60 seconds")
    progress_regex = re.compile(r"Time:\s+([\d.]+)\s+of\s+([\d.]+)\s+seconds")

    # Launch FAST.Farm
    process = subprocess.Popen(
        [fastfarm_exe, fstf_filename],
        cwd=target_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # Merges stderr into stdout stream
        text=True,
        bufsize=1,
    )

    output_lines = []

    # Live console stream reader
    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break

        if line:
            output_lines.append(line)
            match = progress_regex.search(line)
            if match and progress_callback:
                current_time = float(match.group(1))
                total_time = float(match.group(2))
                progress_callback(current_time, total_time)

    leftover_stdout, _ = process.communicate()
    if leftover_stdout:
        output_lines.append(leftover_stdout)

    full_stdout = "".join(output_lines)

    if process.returncode != 0:
        raise RuntimeError(
            f"FAST.Farm exited with error code {process.returncode}\n\n"
            f"{'-'*30}\n"
            f"{full_stdout}\n"
            f"{'-'*30}\n"
            f"Note: Stderr was merged into stdout above."
        )

    # --------------------------------------------------------------------------
    # SNAPSHOT 2: Isolate and move ONLY matching targets
    # --------------------------------------------------------------------------
    post_execution_files = set(os.listdir(target_dir))

    files_to_move = set()

    # Iterate through target directory and apply filter logic
    for file in post_execution_files:
        # Keep JSON files
        if fnmatch.fnmatch(file, "*.json"):
            files_to_move.add(file)
            continue

        # Match against explicit pattern rules
        for pattern in save_patterns:
            if pattern.search(file):
                files_to_move.add(file)
                break

    moved_files = []
    for file in files_to_move:
        source_file_path = os.path.join(target_dir, file)

        if os.path.isfile(source_file_path):
            destination_file_path = os.path.join(final_output_dir, file)
            shutil.move(source_file_path, destination_file_path)
            moved_files.append(destination_file_path)

    return moved_files


def run_turbsim(inp_path, save_dir=".", custom_name=None, progress_callback=None):
    """
    Runs a TurbSim simulation using a live stream reader to capture stdout.
    Optionally reports completion tracking to the progress callback.
    """
    # Look for TurbSim.exe parallel to where OpenFAST.exe is expected
    turbsim_exe = os.path.abspath("TurbSim.exe")
    if not os.path.exists(turbsim_exe):
        raise FileNotFoundError(f"Could not find local executable at: {turbsim_exe}")

    target_inp_path = os.path.abspath(inp_path)
    if not os.path.exists(target_inp_path):
        raise FileNotFoundError(f"The TurbSim file '{target_inp_path}' does not exist.")

    target_dir = os.path.dirname(target_inp_path)
    inp_filename = os.path.basename(target_inp_path)
    inp_base_name = os.path.splitext(inp_filename)[0]

    os.makedirs(save_dir, exist_ok=True)

    # Regex to extract progress tracking variables from TurbSim console text
    # Standard logs look like: "Processing time step    20 of   120"
    progress_regex = re.compile(
        r"Processing\s+(?:grid\s+)?time\s+step\s+(\d+)\s+of\s+(\d+)"
    )

    # Launch TurbSim using Popen matching the OpenFAST runtime pattern
    process = subprocess.Popen(
        [turbsim_exe, inp_filename],
        cwd=target_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    # Continuously parse the live command window outputs
    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break

        if line:
            # Match internal time step flags
            match = progress_regex.search(line)
            if match and progress_callback:
                current_step = float(match.group(1))
                total_steps = float(match.group(2))

                # Send numerical variables back to the active log-friendly tracker
                progress_callback(current_step, total_steps)

    if process.returncode != 0:
        raise RuntimeError(f"TurbSim exited with error code {process.returncode}")

    # File moving routine tailored for TurbSim wind file artifacts (.bts, .wnd, .sum)
    moved_files = []
    turb_extensions = (".bts", ".wnd", ".sum")

    for file in os.listdir(target_dir):
        if file.startswith(inp_base_name) and file.endswith(turb_extensions):
            source_file_path = os.path.join(target_dir, file)

            # Determine renaming structural logic
            if custom_name:
                ext = os.path.splitext(file)[1]
                final_name = f"{custom_name}{ext}"
            else:
                final_name = file

            destination_file_path = os.path.join(save_dir, final_name)

            # Move out of working directory into long-term target storage directory
            shutil.move(source_file_path, destination_file_path)
            moved_files.append(destination_file_path)

    return moved_files
