import os
import sys

# Critical Windows Environment & DLL Initialization for PyTorch / C++ extensions
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if sys.platform == "win32":
    known_dll_dirs = [
        r"C:\ProgramData\anaconda3\Library\bin",
        r"C:\ProgramData\anaconda3\DLLs",
        r"C:\Users\prane\AppData\Roaming\Python\Python310\site-packages\torch\lib",
    ]
    for d in known_dll_dirs:
        if os.path.exists(d):
            try:
                os.add_dll_directory(d)
            except Exception:
                pass


def init_dll_paths() -> None:
    """Helper function to ensure DLL directories are loaded."""
    pass
