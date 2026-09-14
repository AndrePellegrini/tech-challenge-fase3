"""Garante que `pytest tests` funcione a partir da raiz do repositorio.

Sem este arquivo, `pytest tests` falha com ModuleNotFoundError: No module named 'src',
porque o diretorio raiz nao entra em sys.path. Somente `python -m pytest` funcionava.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
