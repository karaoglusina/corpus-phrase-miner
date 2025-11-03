"""
Package initializer for `src`.

Keep this lightweight and avoid importing heavy dependencies or specific symbols.
Expose submodules for convenience.
"""

# Re-export submodules for convenience (without importing their heavy deps here)
from . import text_utils, ngram_extraction, mutual_information, mi_visualization

__all__ = [
    "text_utils",
    "ngram_extraction",
    "mutual_information",
    "mi_visualization",
]