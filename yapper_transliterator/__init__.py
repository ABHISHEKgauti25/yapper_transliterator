"""
yapper_transliterator — a map-first Devanagari -> Roman transliteration library.

A curated dictionary (``yapper_map``) resolves the common case; the compact
``Lipi`` CTC model is the out-of-vocabulary fallback.

    from yapper_transliterator import Transliterator
    t = Transliterator("models/lipi", "data/yapper_map.json", backend="map_lipi")
    t.transliterate_text("नमस्ते दुनिया")
"""

from .lipi import Lipi, LipiConfig, count_parameters
from .transliterator import BACKENDS, Transliterator

__version__ = "0.1.0"

__all__ = [
    "Transliterator",
    "BACKENDS",
    "Lipi",
    "LipiConfig",
    "count_parameters",
    "__version__",
]
