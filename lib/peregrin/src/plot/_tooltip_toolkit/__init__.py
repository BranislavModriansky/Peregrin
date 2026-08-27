from functools import lru_cache
from pathlib import Path


_TOOLKIT_DIR = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def _load_tooltip_assets():
    return {
        'css': (_TOOLKIT_DIR / '_tooltip_styling.css').read_text(encoding='utf-8'),
        'js': (_TOOLKIT_DIR / '_tooltip.js').read_text(encoding='utf-8'),
    }


tooltip_assets = _load_tooltip_assets()

__all__ = ['tooltip_assets']