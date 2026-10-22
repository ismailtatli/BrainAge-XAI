"""Local single-MRI integration around the immutable scientific core."""

import os
import tempfile
from pathlib import Path

# Nonparticipant font caches may persist; uploaded MRI workspaces never do.
_cache = Path(tempfile.gettempdir()) / "brainage-xai-font-cache"
_cache.mkdir(exist_ok=True, mode=0o700)
os.environ.setdefault("MPLCONFIGDIR", str(_cache))
os.environ.setdefault("XDG_CACHE_HOME", str(_cache / "system"))

APPLICATION_VERSION = "0.1.0-local"
