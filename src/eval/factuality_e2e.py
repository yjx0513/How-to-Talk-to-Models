from __future__ import annotations

import re
from typing import Dict, List, Tuple


def _normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def slot_error_rate(mr: Dict[str, str], text: str) -> Dict[str, float]:
    """
    Lightweight, reproducible slot error rate proxy.
    We count a slot as "covered" if the slot value appears as a substring (case-insensitive).
    This is not perfect, but it's simple and fully reproducible.
    """
    t = _normalize(text)
    missing = 0
    covered = 0
    total = 0
    for k, v in mr.items():
        if v is None:
            continue
        total += 1
        if _normalize(str(v)) in t:
            covered += 1
        else:
            missing += 1
    ser = (missing / total) if total > 0 else 0.0
    return {"slots_total": float(total), "slots_covered": float(covered), "slots_missing": float(missing), "ser_missing": float(ser)}

