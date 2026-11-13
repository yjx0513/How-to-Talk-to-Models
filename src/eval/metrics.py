from __future__ import annotations

from typing import Dict, List

import sacrebleu
from nltk.translate.meteor_score import meteor_score as nltk_meteor_score
from rouge_score import rouge_scorer


def _ensure_nltk_data() -> None:
    import nltk

    for resource, pkg in (("corpora/wordnet", "wordnet"), ("corpora/omw-1.4", "omw-1.4")):
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(pkg, quiet=True)


def _corpus_meteor(preds: List[str], refs: List[str]) -> float:
    _ensure_nltk_data()
    scores = [nltk_meteor_score([ref.split()], pred.split()) for pred, ref in zip(preds, refs)]
    return float(sum(scores) / max(1, len(scores)))


def _corpus_rouge(preds: List[str], refs: List[str]) -> Dict[str, float]:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    keys = ("rouge1", "rouge2", "rougeL")
    totals = {k: 0.0 for k in keys}
    for pred, ref in zip(preds, refs):
        s = scorer.score(ref, pred)
        for k in keys:
            totals[k] += s[k].fmeasure
    n = max(1, len(preds))
    return {k: totals[k] / n for k in keys}


def compute_text_metrics(preds: List[str], refs: List[str], enable_bertscore: bool = True) -> Dict[str, float]:
    bleu_score = sacrebleu.corpus_bleu(preds, [[r] for r in refs])
    rouge = _corpus_rouge(preds, refs)

    out = {
        "bleu": float(bleu_score.score),
        "meteor": _corpus_meteor(preds, refs),
        "rouge1": float(rouge["rouge1"]),
        "rouge2": float(rouge["rouge2"]),
        "rougeL": float(rouge["rougeL"]),
    }

    if enable_bertscore:
        import bert_score

        _, _, f1 = bert_score.score(preds, refs, lang="en", verbose=False)
        out["bertscore_f1"] = float(f1.mean())

    return out
