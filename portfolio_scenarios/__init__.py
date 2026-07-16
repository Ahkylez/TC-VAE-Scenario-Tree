"""Portfolio scenario-generation layer built on the upstream ``tsvae`` package.

The upstream Time-Causal VAE lives under ``src/tsvae`` and is imported with bare
``import tsvae`` statements. Importing this package puts ``<repo>/src`` on
``sys.path`` so those imports resolve regardless of the working directory,
letting the scenario-generation code and the training notebook use
``import tsvae`` and ``import portfolio_scenarios.*`` side by side.
"""

import os as _os
import sys as _sys

_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_SRC = _os.path.join(_REPO_ROOT, "src")
if _os.path.isdir(_SRC) and _SRC not in _sys.path:
    _sys.path.insert(0, _SRC)
