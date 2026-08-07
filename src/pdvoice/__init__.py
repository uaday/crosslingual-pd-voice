"""pdvoice — voice-based Parkinson's disease detection (cross-dataset, multilingual).

Keep this package init light: submodules pull in heavy deps (torch, transformers,
librosa) and should be imported explicitly, e.g. `from pdvoice.features import ssl`.
"""

__version__ = "0.1.0"
