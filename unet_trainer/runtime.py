"""Process-local native runtime selection; no machine settings are changed."""
import importlib.util
import os
from pathlib import Path
import sys


def ensure_native_runtime():
    # macOS XGBoost uses @rpath/libomp, while Torch bundles another libomp.
    # dyld must select the same library BEFORE either backend is imported.
    if sys.platform != 'darwin':
        return
    spec = importlib.util.find_spec('torch')
    if spec is None or not spec.origin:
        return
    directory = str(Path(spec.origin).parent / 'lib')
    if not (Path(directory) / 'libomp.dylib').is_file():
        return
    existing = os.environ.get('DYLD_LIBRARY_PATH', '')
    if existing.split(os.pathsep)[0] == directory or os.environ.get('SEGMENTATION_NATIVE_RUNTIME') == directory:
        return
    environment = dict(os.environ)
    environment['DYLD_LIBRARY_PATH'] = directory + (os.pathsep + existing if existing else '')
    environment['SEGMENTATION_NATIVE_RUNTIME'] = directory
    os.execve(sys.executable, [sys.executable, *sys.orig_argv[1:]], environment)
