from __future__ import annotations

import re
from typing import Dict, List, Sequence, Tuple


def _normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


Triple = Tuple[str, str, str]


def triple_coverage(triples: Sequence[Triple], text: str) -> Dict[str, float]:
    """
    Lightweight coverage proxy:
    A triple is considered covered if BOTH subject and object appear in the text.
    """
    t = _normalize(text)
    total = len(triples)
    covered = 0
    for s, r, o in triples:
        if _normalize(s) in t and _normalize(o) in t:
            covered += 1
    cov = (covered / total) if total > 0 else 0.0
    return {"triples_total": float(total), "triples_covered": float(covered), "triple_coverage": float(cov)}

