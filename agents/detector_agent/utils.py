import os
import sys

# Critical Windows Environment & DLL Initialization for PyTorch / C++ extensions
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def _torch_lib_dirs() -> list:
    """Derive torch's native lib directories dynamically instead of hardcoding
    machine-specific paths (which point at other developers' machines and break
    in CI / fresh installs)."""
    if sys.platform != "win32":
        return []
    try:
        import torch
    except Exception:
        return []
    pkg_dir = os.path.dirname(torch.__file__)
    dirs = [pkg_dir]
    lib_dir = os.path.join(pkg_dir, "lib")
    if os.path.exists(lib_dir):
        dirs.append(lib_dir)
    return dirs


if sys.platform == "win32":
    for d in _torch_lib_dirs():
        if os.path.exists(d):
            try:
                os.add_dll_directory(d)
            except Exception:
                pass


def init_dll_paths() -> None:
    """Helper function to ensure DLL directories are loaded."""
    if sys.platform == "win32":
        for d in _torch_lib_dirs():
            if os.path.exists(d):
                try:
                    os.add_dll_directory(d)
                except Exception:
                    pass
