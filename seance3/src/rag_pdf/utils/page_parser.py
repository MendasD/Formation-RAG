"""
Parseur de plages de pages pour la sélection d'ingestion.

Permet à l'utilisateur de saisir une notation compacte du type :
    "1-5, 10, 15-20"
et de récupérer une liste de pages explicite [1, 2, 3, 4, 5, 10, 15, 16, 17, 18, 19, 20].

Supporte :
- Plages : "1-5"
- Pages individuelles : "10"
- Combinaisons (intervalles disjoints) : "1-5, 10, 15-20"
- Espaces tolérés autour des séparateurs.
"""
from __future__ import annotations


class PageRangeError(ValueError):
    """Erreur de parsing d'une expression de plages de pages."""


def parse_page_ranges(expr: str, max_page: int | None = None) -> list[int]:
    """Parse une expression du style "1-5, 10, 15-20" en liste de pages triées et dédupliquées.

    Args:
        expr: La chaîne à parser. Vide ou None → retourne [] (= toutes les pages côté caller).
        max_page: Si fourni, vérifie que toutes les pages demandées sont ≤ max_page.

    Returns:
        Liste triée et dédupliquée de numéros de page (1-indexés).

    Raises:
        PageRangeError: Si l'expression est mal formée ou contient des pages invalides.

    Examples:
        >>> parse_page_ranges("1-3, 5, 7-8")
        [1, 2, 3, 5, 7, 8]
        >>> parse_page_ranges("")
        []
        >>> parse_page_ranges("5-3")
        Traceback (most recent call last):
            ...
        PageRangeError: Plage inversée : '5-3'
    """
    if not expr or not expr.strip():
        return []

    pages: set[int] = set()

    for token in expr.split(","):
        token = token.strip()
        if not token:
            continue

        if "-" in token:
            # Plage du style "1-5".
            parts = [p.strip() for p in token.split("-")]
            if len(parts) != 2 or not all(parts):
                raise PageRangeError(f"Plage invalide : {token!r}")
            try:
                start, end = int(parts[0]), int(parts[1])
            except ValueError:
                raise PageRangeError(f"Plage non numérique : {token!r}")
            if start < 1 or end < 1:
                raise PageRangeError(f"Pages négatives interdites : {token!r}")
            if start > end:
                raise PageRangeError(f"Plage inversée : {token!r}")
            pages.update(range(start, end + 1))
        else:
            # Page unique du style "10".
            try:
                page = int(token)
            except ValueError:
                raise PageRangeError(f"Numéro invalide : {token!r}")
            if page < 1:
                raise PageRangeError(f"Page négative interdite : {token!r}")
            pages.add(page)

    result = sorted(pages)

    if max_page is not None:
        invalid = [p for p in result if p > max_page]
        if invalid:
            raise PageRangeError(
                f"Pages hors document (PDF de {max_page} pages) : {invalid}"
            )

    return result


def format_pages_compact(pages: list[int]) -> str:
    """Inverse de parse_page_ranges : convertit une liste de pages en notation compacte.

    Examples:
        >>> format_pages_compact([1, 2, 3, 5, 7, 8])
        '1-3, 5, 7-8'
    """
    if not pages:
        return ""
    pages = sorted(set(pages))
    ranges: list[str] = []
    start = end = pages[0]
    for p in pages[1:]:
        if p == end + 1:
            end = p
        else:
            ranges.append(f"{start}-{end}" if start != end else str(start))
            start = end = p
    ranges.append(f"{start}-{end}" if start != end else str(start))
    return ", ".join(ranges)
