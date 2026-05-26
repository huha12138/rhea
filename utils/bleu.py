"""BLEU metric wrapper used by score aggregation."""

from typing import Dict, List

import sacrebleu


def compute(
    predictions: List[str],
    references: List[List[str]],
    lang: str = "en",
) -> Dict[str, float]:
    """Compute corpus BLEU with a Hugging Face-style return shape."""
    del lang
    if not predictions:
        return {"bleu": 0.0}

    if references and isinstance(references[0], str):
        formatted_references = [references]
    else:
        formatted_references = [
            [row[i] if i < len(row) else "" for row in references]
            for i in range(max(len(row) for row in references))
        ]

    score = sacrebleu.corpus_bleu(predictions, formatted_references)
    return {"bleu": score.score}
