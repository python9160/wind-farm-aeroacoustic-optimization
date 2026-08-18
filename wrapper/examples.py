# Opening a FST file:
from . import OpenFASTFile, run_openfast

# Opening a new version of base.fst called test_1
wt = OpenFASTFile(r"5MW_farm\base.fst", "test_1")

# Note: OpenFASTFile's entries are auto-populated into _ipython_key_completions_, so typing wt. in a REPL environment such as jupyter will automatically list all the values in the file. Some complex forms like tables are not built to be edited with this simple wrapper.

# Opening the AeroDyn file as an OpenFASTFile object
# This is automatically named "test_1.AeroFile"
aero = wt.AeroFile.open()

# Linking the new file back to the test_1 object
# This resolves the path back to a string
wt.AeroFile.link(aero)

# .ol() automatically opens and links the files:
aero = wt.AeroFile.ol()

# Example modification
aero.MaxIter = 100

# Saving file:
# Saves to test_1.AeroFile.dat
aero.toFile()

# Saves to test_1.fst
wt.toFile()

# Running a file:
new_files = run_openfast(wt)  # Saves files to ./test_1/
new_files = run_openfast(wt, "subfolder")  # Saves files to ./subfolder/test_1
new_files = run_openfast(wt, "subfolder", "trial0")  # Saves files to ./subfolder/trial0


def callback(current, total):
    print(
        f"{current} / {total}"
    )  # works well with tqdm, just initialize pbar with total=float(wt.TMax)


new_files = run_openfast(
    wt, "subfolder", "trial0", callback
)  # Saves files to ./subfolder/trial0 whilst showing live progress
