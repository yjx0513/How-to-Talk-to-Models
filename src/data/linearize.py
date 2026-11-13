from __future__ import annotations

import random
from typing import Dict, Iterable, List, Sequence, Tuple


def _sorted_items_canonical(mr: Dict[str, str]) -> List[Tuple[str, str]]:
    # Canonical field order for E2E.
    order = [
        "name",
        "eatType",
        "food",
        "priceRange",
        "customer rating",
        "area",
        "familyFriendly",
        "near",
    ]
    present = {k: v for k, v in mr.items()}
    items: List[Tuple[str, str]] = []
    for k in order:
        if k in present:
            items.append((k, present[k]))
    for k in sorted(present.keys()):
        if k not in {x for x, _ in items}:
            items.append((k, present[k]))
    return items


def linearize_e2e(mr: Dict[str, str], strategy: str, canonical_missing: str = "none") -> str:
    """
    mr: slot -> value
    strategy: plain | canonical | prompt | tagged
    canonical_missing: none | skip
    """

    strategy = strategy.lower()
    if strategy not in {"plain", "canonical", "prompt", "tagged"}:
        raise ValueError(f"Unknown strategy: {strategy}")
    '''area=riverside ; eatType=coffee shop ; food=French ; name=The Eagle ; priceRange=<none>'''
    if strategy == "canonical":
        items = _sorted_items_canonical(mr)
    #name=The Eagle ; eatType=coffee shop ; food=French ; area=riverside ; priceRange=<none>
    else:
        items = list(mr.items())

    if strategy in {"plain", "canonical"}:
        parts: List[str] = []
        for k, v in items:
            if v is None:
                if canonical_missing == "skip":
                    continue
                v = "<none>"
            parts.append(f"{k}={v}")
        return " ; ".join(parts)
    '''Generate a restaurant description with: name The Eagle, eatType coffee shop, food French, area riverside, priceRange None'''
    if strategy == "prompt":
        parts = [f"{k} {v}" for k, v in items]
        return "Generate a restaurant description with: " + ", ".join(parts)
    '''<name>The Eagle</name> <eatType>coffee shop</eatType> <food>French</food> <area>riverside</area> <priceRange>None</priceRange>'''
    # tagged
    parts = [f"<{k}>{v}</{k}>" for k, v in items]
    return " ".join(parts)


Triple = Tuple[str, str, str]


def linearize_webnlg(
    triples: Sequence[Triple],
    strategy: str,
    triple_order: str = "canonical",
    rng: random.Random | None = None,
) -> str:
    """
    triples: list of (subj, rel, obj)
    strategy: plain | canonical | prompt | tagged
    triple_order: canonical | random
    """
    strategy = strategy.lower()
    if strategy not in {"plain", "canonical", "prompt", "tagged"}:
        raise ValueError(f"Unknown strategy: {strategy}")

    triple_order = triple_order.lower()
    triples_list = list(triples)

    if triple_order == "random":
        (rng or random).shuffle(triples_list)
    elif triple_order == "canonical":
        triples_list = sorted(triples_list, key=lambda t: (t[1], t[0], t[2]))
    else:
        raise ValueError(f"Unknown triple_order: {triple_order}")

    if strategy in {"plain", "canonical"}:
        # Keep identical text form for "plain" vs "canonical" on WebNLG;
        # the difference is controlled via triple_order.
        parts = [f"({s}, {r}, {o})" for s, r, o in triples_list]
        return " ; ".join(parts)

    if strategy == "prompt":
        parts = [f"subject {s}, relation {r}, object {o}" for s, r, o in triples_list]
        return "Generate a description that covers: " + " ; ".join(parts)

    # tagged
    parts = [f"<subj>{s}</subj> <rel>{r}</rel> <obj>{o}</obj>" for s, r, o in triples_list]
    return " <t> ".join(parts)

