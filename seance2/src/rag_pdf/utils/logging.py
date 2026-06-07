"""
Logger structuré utilisant Rich pour un rendu coloré et lisible.

En production réelle on remplacerait souvent par `structlog` + JSON,
mais Rich est parfait pour le développement et les démos.
"""
import logging
from rich.logging import RichHandler


def get_logger(name: str = "rag_pdf", level: int = logging.INFO) -> logging.Logger:
    """Récupère un logger configuré avec Rich.

    Args:
        name: Nom du logger (typiquement `__name__` dans chaque module).
        level: Niveau de log (DEBUG, INFO, WARNING, ERROR).
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = RichHandler(
            rich_tracebacks=True,
            show_path=False,
            show_time=True,
            markup=True,
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger
